import requests
import pandas as pd
from typing import Dict, List, Optional
from io import StringIO
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import itertools
import concurrent.futures
import time
from scipy import stats
from itertools import product

@dataclass
class SimulationRun:
    """Class to store results of a single simulation run"""
    run_id: int
    config: Dict
    final_status: str
    years_survived: int
    final_population: int
    final_resources: float
    final_health: float
    distance_covered: float
    disease_outbreaks: int
    overcrowding_events: int
    critical_rationing_events: int
    normal_rationing_events: int

class EnhancedGenerationShipClient:
    def __init__(self, base_url: str = 'http://localhost:5001'):
        self.base_url = base_url.rstrip('/')
        self.simulation_runs = []
        self.current_simulation_id = None
        
    def run_multiple_simulations(self, config: Dict, num_runs: int = 10, max_years: int = 1000) -> None:
        """
        Run multiple simulations with the same configuration and store results
        
        Args:
            config: Simulation configuration dictionary
            num_runs: Number of simulation runs to perform
            max_years: Maximum number of years to simulate per run
        """
        self.simulation_runs = [] 
        for run_id in range(num_runs):
            print(f"Starting simulation run {run_id + 1}/{num_runs}")
            
            # Initialize new simulation
            self.initialize(config)
            
            # Track events for this run
            disease_outbreaks = 0
            overcrowding_events = 0
            critical_rationing_events = 0
            normal_rationing_events = 0
            
            # Run simulation year by year until completion or max_years
            final_status = None
            years_survived = 0
            final_population = 0
            final_resources = 0
            final_health = 0
            distance_covered = 0
            
            for year in range(max_years):
                results = self.simulate(years=1)
                if not results:  # Empty results means simulation failed
                    break
                    
                last_year = results[-1]
                
                # Update event counters
                if last_year.get('diseaseOutbreakEvent', 0):
                    disease_outbreaks += 1
                if last_year.get('overCrowdingEvent', 0):
                    overcrowding_events += 1
                if last_year.get('criticalRationingEvent', 0):
                    critical_rationing_events += 1
                if last_year.get('normalRationingEvent', 0):
                    normal_rationing_events += 1
                
                # Update final statistics
                final_status = last_year['status']
                years_survived = year + 1
                final_population = last_year['population']
                final_resources = last_year['resources']
                final_health = last_year['health_index']
                distance_covered = last_year['distance_covered']
                
                if final_status in ['Success', 'Failed']:
                    break
            
            # Store run results
            run_result = SimulationRun(
                run_id=run_id,
                config=config.copy(),
                final_status=final_status,
                years_survived=years_survived,
                final_population=final_population,
                final_resources=final_resources,
                final_health=final_health,
                distance_covered=distance_covered,
                disease_outbreaks=disease_outbreaks,
                overcrowding_events=overcrowding_events,
                critical_rationing_events=critical_rationing_events,
                normal_rationing_events=normal_rationing_events
            )
            self.simulation_runs.append(run_result)
            
            # Reset simulation for next run
            self.reset()

    def generate_summary_report(self, filename: str = None) -> pd.DataFrame:
        """
        Generate a summary report of all simulation runs
        
        Args:
            filename: Optional filename to save the report (CSV or XLSX)
            
        Returns:
            DataFrame containing the summary report
        """
        if not self.simulation_runs:
            raise ValueError("No simulation runs to summarize")
            
        # Convert simulation runs to DataFrame
        runs_data = []
        for run in self.simulation_runs:
            run_dict = {
                'Run ID': run.run_id,
                'Final Status': run.final_status,
                'Years Survived': run.years_survived,
                'Final Population': run.final_population,
                'Final Resources': run.final_resources,
                'Final Health': run.final_health,
                'Distance Covered (km)': run.distance_covered,
                'Disease Outbreaks': run.disease_outbreaks,
                'Overcrowding Events': run.overcrowding_events,
                'Critical Rationing Events': run.critical_rationing_events,
                'Normal Rationing Events': run.normal_rationing_events,
            }
            # Add configuration parameters
            for key, value in run.config.items():
                run_dict[f'Config_{key}'] = value
            runs_data.append(run_dict)
            
        df = pd.DataFrame(runs_data)
        
        # Add summary statistics
        summary_stats = pd.DataFrame({
            'Metric': [
                'Success Rate',
                'Average Years Survived',
                'Average Final Population',
                'Average Final Resources',
                'Average Final Health',
                'Average Distance Covered',
                'Average Disease Outbreaks',
                'Average Overcrowding Events',
                'Average Critical Rationing Events',
                'Average Normal Rationing Events'
            ],
            'Value': [
                f"{(df['Final Status'] == 'Success').mean() * 100:.1f}%",
                f"{df['Years Survived'].mean():.1f}",
                f"{df['Final Population'].mean():.1f}",
                f"{df['Final Resources'].mean():.1f}",
                f"{df['Final Health'].mean():.1f}",
                f"{df['Distance Covered (km)'].mean():.1f}",
                f"{df['Disease Outbreaks'].mean():.1f}",
                f"{df['Overcrowding Events'].mean():.1f}",
                f"{df['Critical Rationing Events'].mean():.1f}",
                f"{df['Normal Rationing Events'].mean():.1f}"
            ]
        })
        
        # Save to file if filename provided
        if filename:
            if filename.endswith('.xlsx'):
                with pd.ExcelWriter(filename) as writer:
                    df.to_excel(writer, sheet_name='Detailed Results', index=False)
                    summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
            else:  # Save as CSV
                df.to_csv(filename, index=False)
                
        return df, summary_stats

    # Include existing methods from GenerationShipClient
    def initialize(self, config: Dict) -> Dict:
        required_params = {
            'initial_population',
            'ship_capacity',
            'initial_resources',
            'birth_rate',
            'death_rate',
            'resource_gen_rate',
            'lightspeed_fraction',
            'health_index'
        }

        missing_params = required_params - set(config.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        response = requests.post(f"{self.base_url}/initialize", json=config)
        response.raise_for_status()
        result = response.json()
        self.current_simulation_id = result['simulation_id']
        return result

    def simulate(self, years: int = 1) -> List[Dict]:
        if not self.current_simulation_id:
            raise ValueError("No active simulation. Call initialize() first.")
            
        response = requests.post(
            f"{self.base_url}/simulate/{self.current_simulation_id}", 
            json={'years': years}
        )
        response.raise_for_status()
        return response.json()

    def reset(self) -> Dict:
        if not self.current_simulation_id:
            raise ValueError("No active simulation. Call initialize() first.")
            
        response = requests.post(f"{self.base_url}/reset/{self.current_simulation_id}")
        response.raise_for_status()
        return response.json()

    def export_csv(self) -> str:
        if not self.current_simulation_id:
            raise ValueError("No active simulation. Call initialize() first.")
            
        response = requests.get(f"{self.base_url}/export-csv/{self.current_simulation_id}")
        response.raise_for_status()
        return response.content.decode('utf-8')

# Generation Ship Configurations Tuned to Server Mechanics

class DetailedSimulationClient(EnhancedGenerationShipClient):
    def run_detailed_simulation(self, config: Dict, num_runs: int = 10, max_years: int = 1000,
                              detail_years: int = 10) -> None:
        """
        Run simulations with detailed monitoring of early years
        
        Args:
            config: Simulation configuration
            num_runs: Number of simulation runs
            max_years: Maximum simulation years
            detail_years: Number of years to track in detail
        """
        self.simulation_runs = []
        early_year_data = []
        
        for run_id in range(num_runs):
            print(f"\nStarting simulation run {run_id + 1}/{num_runs}")
            self.initialize(config)
            
            # Track events
            disease_outbreaks = 0
            overcrowding_events = 0
            critical_rationing_events = 0
            normal_rationing_events = 0
            
            # Track detailed early years
            yearly_data = []
            
            for year in range(max_years):
                results = self.simulate(years=1)
                if not results:
                    break
                    
                last_year = results[-1]
                
                # Update event counters
                if last_year.get('diseaseOutbreakEvent', 0):
                    disease_outbreaks += 1
                if last_year.get('overCrowdingEvent', 0):
                    overcrowding_events += 1
                if last_year.get('criticalRationingEvent', 0):
                    critical_rationing_events += 1
                if last_year.get('normalRationingEvent', 0):
                    normal_rationing_events += 1
                
                # Store detailed data for early years
                if year < detail_years:
                    yearly_data.append({
                        'run_id': run_id,
                        'year': year,
                        'population': last_year['population'],
                        'resources': last_year['resources'],
                        'health_index': last_year['health_index'],
                        'resource_gen_rate': last_year['resource_gen_rate'],
                        'disease_outbreak': last_year.get('diseaseOutbreakEvent', 0),
                        'overcrowding': last_year.get('overCrowdingEvent', 0),
                        'critical_rationing': last_year.get('criticalRationingEvent', 0),
                        'normal_rationing': last_year.get('normalRationingEvent', 0)
                    })
                
                # Update final statistics
                final_status = last_year['status']
                years_survived = year + 1
                final_population = last_year['population']
                final_resources = last_year['resources']
                final_health = last_year['health_index']
                distance_covered = last_year['distance_covered']
                
                if final_status in ['Success', 'Failed']:
                    break
            
            # Store run results (keeping existing logic)
            run_result = SimulationRun(
                run_id=run_id,
                config=config.copy(),
                final_status=final_status,
                years_survived=years_survived,
                final_population=final_population,
                final_resources=final_resources,
                final_health=final_health,
                distance_covered=distance_covered,
                disease_outbreaks=disease_outbreaks,
                overcrowding_events=overcrowding_events,
                critical_rationing_events=critical_rationing_events,
                normal_rationing_events=normal_rationing_events
            )
            self.simulation_runs.append(run_result)
            early_year_data.extend(yearly_data)
            
            self.reset()
            
        # Convert early year data to DataFrame for analysis
        early_df = pd.DataFrame(early_year_data)
        return early_df

    def analyze_early_years(self, early_df: pd.DataFrame) -> None:
        """Analyze and print statistics about early years"""
        print("\nEarly Years Analysis:")
        print("-" * 50)
        
        # Analyze each year
        for year in early_df['year'].unique():
            year_data = early_df[early_df['year'] == year]
            print(f"\nYear {year}:")
            print(f"Average Population: {year_data['population'].mean():.1f}")
            print(f"Average Resources: {year_data['resources'].mean():.1f}")
            print(f"Average Health: {year_data['health_index'].mean():.1f}")
            print(f"Disease Outbreaks: {year_data['disease_outbreak'].sum()}")
            print(f"Overcrowding Events: {year_data['overcrowding'].sum()}")
            print(f"Critical Rationing Events: {year_data['critical_rationing'].sum()}")
            
        # Calculate survival statistics
        survival_data = early_df.groupby('run_id')['year'].max()
        print("\nSurvival Statistics:")
        print(f"Runs ending in first {len(early_df['year'].unique())} years: "
              f"{(survival_data < early_df['year'].max()).mean() * 100:.1f}%")
        

class StabilityAnalysisClient(EnhancedGenerationShipClient):
    def analyze_stability(self, config: Dict, num_runs: int = 100) -> Dict:
        """
        Analyze the stability of simulation outcomes for a given configuration.
        
        Args:
            config: Simulation configuration dictionary
            num_runs: Number of simulation runs to perform
            
        Returns:
            Dictionary containing variance analysis of key metrics
        """
        # Run simulations
        self.run_multiple_simulations(config, num_runs=num_runs)
        
        # Extract key metrics for analysis
        metrics = {
            'years_survived': [],
            'final_population': [],
            'final_resources': [],
            'final_health': [],
            'distance_covered': [],
            'success_rate': []
        }
        
        # Calculate success rate for each 10% of runs
        batch_size = max(num_runs // 10, 1)
        for i in range(0, num_runs, batch_size):
            batch = self.simulation_runs[i:i+batch_size]
            success_rate = sum(1 for run in batch if run.final_status == "Success") / len(batch)
            metrics['success_rate'].append(success_rate)
        
        # Collect other metrics
        for run in self.simulation_runs:
            metrics['years_survived'].append(run.years_survived)
            metrics['final_population'].append(run.final_population)
            metrics['final_resources'].append(run.final_resources)
            metrics['final_health'].append(run.final_health)
            metrics['distance_covered'].append(run.distance_covered)
                
        analysis = {
            'success_rate': {
                'mean': np.mean(metrics['success_rate']),
                'variance': np.var(metrics['success_rate']),
                'std': np.std(metrics['success_rate'])
            }
        }
        
        for metric in ['years_survived', 'final_population', 'final_resources', 
                      'final_health', 'distance_covered']:
            analysis[metric] = {
                'mean': np.mean(metrics[metric]),
                'variance': np.var(metrics[metric]),
                'std': np.std(metrics[metric]),
                'coefficient_of_variation': np.std(metrics[metric]) / np.mean(metrics[metric])
                if np.mean(metrics[metric]) != 0 else float('inf')
            }
            
        # Calculate stability score (lower means more stable)
        # Weighted average of coefficient of variation for key metrics
        weights = {
            'success_rate': 0.3,
            'years_survived': 0.2,
            'final_population': 0.2,
            'final_resources': 0.15,
            'final_health': 0.15
        }
        
        stability_components = []
        for metric, weight in weights.items():
            if metric == 'success_rate':
                # For success rate, use standard deviation directly
                stability_components.append(weight * analysis[metric]['std'])
            else:
                # For other metrics, use coefficient of variation
                stability_components.append(
                    weight * analysis[metric]['coefficient_of_variation']
                )
        
        analysis['overall_stability_score'] = sum(stability_components)
        
        return analysis

    def print_stability_analysis(self, analysis: Dict) -> None:
        """Pretty print the stability analysis results."""
        print("\nStability Analysis Results")
        print("=" * 50)
        
        print("\nSuccess Rate Statistics:")
        print(f"Mean: {analysis['success_rate']['mean']:.2%}")
        print(f"Standard Deviation: {analysis['success_rate']['std']:.2%}")
        
        metrics = ['years_survived', 'final_population', 'final_resources', 
                  'final_health', 'distance_covered']
        
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()} Statistics:")
            print(f"Mean: {analysis[metric]['mean']:.2f}")
            print(f"Standard Deviation: {analysis[metric]['std']:.2f}")
            print(f"Coefficient of Variation: {analysis[metric]['coefficient_of_variation']:.2%}")
        
        print("\nOverall Stability Score:", f"{analysis['overall_stability_score']:.4f}")
        print("(Lower score indicates more stable configuration)")

# Generation Ship Configurations with Balanced Challenges

configurations = {
    "steady_pressure": {
        "initial_population": 1000,     
        "ship_capacity": 1100,          
        "initial_resources": 110000,    
        "birth_rate": 9.4,              
        "death_rate": 3.7,               
        "resource_gen_rate": 99.7,        
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    },
    "overcrowding_risk": {
        "initial_population": 1000,     
        "ship_capacity": 1100,          
        "initial_resources": 110000,    
        "birth_rate": 9.4,              
        "death_rate": 3.7,               
        "resource_gen_rate": 100.5,        
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    },
        "overcrowding_risk_2": {
        "initial_population": 1000,     
        "ship_capacity": 1050,          
        "initial_resources": 110000,    
        "birth_rate": 9.4,              
        "death_rate": 3.7,               
        "resource_gen_rate": 103.5,        
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    }
}


class ShipWeightOptimizer:
    def __init__(self):
        self.client = StabilityAnalysisClient()
        
    def evaluate_configuration(self, capacity, resources, resource_gen_rate, num_runs=50):
        config = {
            "initial_population": 1000,  # Fixed initial population
            "ship_capacity": capacity,
            "initial_resources": resources,
            "birth_rate": 9.4,              
            "death_rate": 3.7,               
            "resource_gen_rate": resource_gen_rate,        
            "lightspeed_fraction": 0.0059,
            "health_index": 100
        }
        
        # Run stability analysis
        stability = self.client.analyze_stability(config, num_runs)
        
        # Calculate efficiency metrics
        ship_weight = capacity * 100 + resources  # Unit weight for capacity and resources
        
        return {
            'config': config,
            'ship_weight': ship_weight,
            'final_population_mean': stability['final_population']['mean'],
            'final_population_std': stability['final_population']['std'],
            'population_efficiency': stability['final_population']['mean'] / ship_weight,
            'years_survived_mean': stability['years_survived']['mean']
        }

class ParallelGenerationShipClient:
    def __init__(self, base_url: str = 'http://localhost:5001', max_workers: int = 4):
        self.base_url = base_url.rstrip('/')
        self.max_workers = max_workers
        self.session = requests.Session()
        self.simulation_runs = []

    def _run_single_simulation(self, run_id: int, config: Dict, max_years: int = 1000) -> SimulationRun:
        """Run a single simulation with error handling and retries"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Initialize simulation
                response = self.session.post(f"{self.base_url}/initialize", json=config)
                response.raise_for_status()
                simulation_id = response.json()['simulation_id']

                # Run simulation for all years at once
                response = self.session.post(
                    f"{self.base_url}/simulate/{simulation_id}", 
                    json={'years': max_years}
                )
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    raise Exception("Simulation failed with no results")
                    
                # Get final year results
                last_year = results[-1]
                
                # Count total events across all years
                disease_outbreaks = sum(year.get('diseaseOutbreakEvent', 0) for year in results)
                overcrowding_events = sum(year.get('overCrowdingEvent', 0) for year in results)
                critical_rationing_events = sum(year.get('criticalRationingEvent', 0) for year in results)
                normal_rationing_events = sum(year.get('normalRationingEvent', 0) for year in results)
                
                return SimulationRun(
                    run_id=run_id,
                    config=config.copy(),
                    final_status=last_year['status'],
                    years_survived=len(results),
                    final_population=last_year['population'],
                    final_resources=last_year['resources'],
                    final_health=last_year['health_index'],
                    distance_covered=last_year['distance_covered'],
                    disease_outbreaks=disease_outbreaks,
                    overcrowding_events=overcrowding_events,
                    critical_rationing_events=critical_rationing_events,
                    normal_rationing_events=normal_rationing_events
                )
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed to complete simulation {run_id} after {max_retries} attempts: {str(e)}")
                    raise
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue

    def run_parallel_simulations(self, config: Dict, num_runs: int = 10, max_years: int = 1000) -> List[SimulationRun]:
        """Run multiple simulations in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all simulation runs
            future_to_run = {
                executor.submit(self._run_single_simulation, i, config, max_years): i 
                for i in range(num_runs)
            }
            
            # Process results as they complete
            completed_runs = []
            for future in concurrent.futures.as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    result = future.result()
                    completed_runs.append(result)
                    print(f"Completed simulation run {run_id + 1}/{num_runs}")
                except Exception as e:
                    print(f"Simulation run {run_id} generated an exception: {str(e)}")
            
            self.simulation_runs = completed_runs
            return completed_runs

    def analyze_results(self) -> pd.DataFrame:
        """Analyze simulation results and return a DataFrame"""
        if not self.simulation_runs:
            raise ValueError("No simulation runs to analyze")
            
        runs_data = []
        for run in self.simulation_runs:
            run_dict = {
                'Run ID': run.run_id,
                'Final Status': run.final_status,
                'Years Survived': run.years_survived,
                'Final Population': run.final_population,
                'Final Resources': run.final_resources,
                'Final Health': run.final_health,
                'Distance Covered (km)': run.distance_covered,
                'Disease Outbreaks': run.disease_outbreaks,
                'Overcrowding Events': run.overcrowding_events,
                'Critical Rationing Events': run.critical_rationing_events,
                'Normal Rationing Events': run.normal_rationing_events
            }
            # Add configuration parameters
            for key, value in run.config.items():
                run_dict[f'Config_{key}'] = value
            runs_data.append(run_dict)
            
        return pd.DataFrame(runs_data)

    def get_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for all simulation runs"""
        df = self.analyze_results()
        
        summary_stats = pd.DataFrame({
            'Metric': [
                'Success Rate',
                'Average Years Survived',
                'Average Final Population',
                'Average Final Resources',
                'Average Final Health',
                'Average Distance Covered',
                'Average Disease Outbreaks',
                'Average Overcrowding Events',
                'Average Critical Rationing Events',
                'Average Normal Rationing Events'
            ],
            'Value': [
                f"{(df['Final Status'] == 'Success').mean() * 100:.1f}%",
                f"{df['Years Survived'].mean():.1f}",
                f"{df['Final Population'].mean():.1f}",
                f"{df['Final Resources'].mean():.1f}",
                f"{df['Final Health'].mean():.1f}",
                f"{df['Distance Covered (km)'].mean():.1f}",
                f"{df['Disease Outbreaks'].mean():.1f}",
                f"{df['Overcrowding Events'].mean():.1f}",
                f"{df['Critical Rationing Events'].mean():.1f}",
                f"{df['Normal Rationing Events'].mean():.1f}"
            ]
        })
        
        return summary_stats
    

class EnhancedShipWeightOptimizer:
    def __init__(self, client):
        self.client = client
        self.results_cache = []
        
    def evaluate_configuration(self, capacity: int, resources: float, 
                             resource_gen_rate: float, num_runs: int = 50) -> Dict:
        """
        Evaluate a specific configuration with statistical analysis
        """
        config = {
            "initial_population": 1000,  # Fixed
            "ship_capacity": capacity,
            "initial_resources": resources,
            "birth_rate": 9.4,           # Fixed
            "death_rate": 3.7,           # Fixed
            "resource_gen_rate": resource_gen_rate,
            "lightspeed_fraction": 0.0059,  # Fixed
            "health_index": 100          # Fixed
        }
        
        # Run parallel simulations
        self.client.run_parallel_simulations(config, num_runs=num_runs)
        results_df = self.client.analyze_results()
        
        # Calculate ship weight
        ship_weight = capacity * 100 + resources  # Assumed weight formula
        
        # Statistical analysis
        final_pop_stats = {
            'mean': results_df['Final Population'].mean(),
            'std': results_df['Final Population'].std(),
            'ci_lower': stats.t.interval(0.95, len(results_df)-1, 
                                       loc=results_df['Final Population'].mean(),
                                       scale=stats.sem(results_df['Final Population']))[0],
            'ci_upper': stats.t.interval(0.95, len(results_df)-1,
                                       loc=results_df['Final Population'].mean(),
                                       scale=stats.sem(results_df['Final Population']))[1]
        }
        
        # Calculate efficiency metrics
        efficiency_metrics = {
            'pop_per_weight': final_pop_stats['mean'] / ship_weight,
            'pop_per_weight_ci_lower': final_pop_stats['ci_lower'] / ship_weight,
            'pop_per_weight_ci_upper': final_pop_stats['ci_upper'] / ship_weight
        }
        
        result = {
            'config': config,
            'ship_weight': ship_weight,
            'final_population': final_pop_stats,
            'efficiency': efficiency_metrics,
            'success_rate': (results_df['Final Status'] == 'Success').mean()
        }
        
        self.results_cache.append(result)
        return result
        
    def grid_search(self,
                   capacities: List[int],
                   resources: List[float],
                   resource_gen_rates: List[float],
                   num_runs: int = 50) -> pd.DataFrame:
        """
        Perform grid search over parameter space
        """
        results = []
        total_combinations = len(capacities) * len(resources) * len(resource_gen_rates)
        current = 0
        
        for capacity, resource, rate in product(capacities, resources, resource_gen_rates):
            current += 1
            print(f"Evaluating combination {current}/{total_combinations}")
            
            result = self.evaluate_configuration(capacity, resource, rate, num_runs)
            results.append({
                'capacity': capacity,
                'initial_resources': resource,
                'resource_gen_rate': rate,
                'ship_weight': result['ship_weight'],
                'final_pop_mean': result['final_population']['mean'],
                'final_pop_std': result['final_population']['std'],
                'final_pop_ci_lower': result['final_population']['ci_lower'],
                'final_pop_ci_upper': result['final_population']['ci_upper'],
                'pop_per_weight': result['efficiency']['pop_per_weight'],
                'success_rate': result['success_rate']
            })
            
        return pd.DataFrame(results)

    def analyze_pareto_frontier(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Pareto-optimal configurations (maximizing final population, minimizing weight)
        """
        pareto_optimal = []
        for idx, row in results_df.iterrows():
            dominated = False
            for idx2, row2 in results_df.iterrows():
                if (idx != idx2 and 
                    row2['final_pop_mean'] >= row['final_pop_mean'] and
                    row2['ship_weight'] <= row['ship_weight'] and
                    (row2['final_pop_mean'] > row['final_pop_mean'] or 
                     row2['ship_weight'] < row['ship_weight'])):
                    dominated = True
                    break
            if not dominated:
                pareto_optimal.append(row)
        
        return pd.DataFrame(pareto_optimal)

# 1. Initialize the optimizer with parallel processing
client = ParallelGenerationShipClient(max_workers=4)
optimizer = EnhancedShipWeightOptimizer(client)

# 2. Define parameter ranges for grid search
capacities = [1050, 1100, 1150, 1200, 1250]  # Ship capacities
resources = [100000, 110000, 120000, 130000]  # Initial resources
resource_gen_rates = [95.0, 97.5, 100.0, 102.5, 105.0]  # Resource generation rates

# 3. Run grid search
results_df = optimizer.grid_search(
    capacities=capacities,
    resources=resources,
    resource_gen_rates=resource_gen_rates,
    num_runs=50  # 50 runs per configuration for statistical significance
)

# 4. Find Pareto-optimal configurations
pareto_df = optimizer.analyze_pareto_frontier(results_df)

# 5. Save results
results_df.to_csv('optimization_results.csv')
pareto_df.to_csv('pareto_optimal_configs.csv')