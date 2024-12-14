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

        print(config)  # Before the requests.post call
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

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CRNComparisonResult:
    """Store results of CRN comparison analysis"""
    config_a: Dict
    config_b: Dict
    crn_differences: pd.DataFrame  # Paired differences with CRN
    non_crn_differences: pd.DataFrame  # Unpaired differences without CRN
    variance_reduction: Dict  # Variance reduction ratios for different metrics
    statistical_tests: Dict  # Results of statistical tests

class CRNComparisonClient(ParallelGenerationShipClient):
    def __init__(self, base_url: str = 'http://localhost:5001', max_workers: int = 4):
        super().__init__(base_url, max_workers)
        
    def run_crn_comparison(self, config_a: Dict, config_b: Dict, 
                          num_runs: int = 50, crn_base_seed: int = 42) -> CRNComparisonResult:
        """
        Run comparative analysis with and without CRN
        
        Args:
            config_a: First configuration to test
            config_b: Second configuration to test
            num_runs: Number of simulation runs
            crn_base_seed: Base seed for CRN runs
        """
        # Run simulations with CRN (paired)
        crn_results_a = []
        crn_results_b = []
        
        print("\nRunning CRN paired simulations...")
        for i in range(num_runs):
            # Use different seeds for each pair but same seed within pair
            current_seed = crn_base_seed + i
            
            # Run config A with CRN
            response = self.session.post(f"{self.base_url}/initialize", 
                                       json={"config": config_a, "crn_seed": current_seed})
            simulation_id = response.json()['simulation_id']
            results = self.session.post(f"{self.base_url}/simulate/{simulation_id}", 
                                      json={'years': 1000}).json()
            crn_results_a.append(self._extract_metrics(results[-1]))
            
            # Run config B with same CRN seed
            response = self.session.post(f"{self.base_url}/initialize", 
                                       json={"config": config_b, "crn_seed": current_seed})
            simulation_id = response.json()['simulation_id']
            results = self.session.post(f"{self.base_url}/simulate/{simulation_id}", 
                                      json={'years': 1000}).json()
            crn_results_b.append(self._extract_metrics(results[-1]))
            
            print(f"Completed CRN pair {i+1}/{num_runs}")
        
        # Run simulations without CRN (unpaired)
        print("\nRunning non-CRN simulations...")
        non_crn_results_a = self._run_non_crn_simulations(config_a, num_runs)
        non_crn_results_b = self._run_non_crn_simulations(config_b, num_runs)
        
        # Convert results to DataFrames
        crn_df_a = pd.DataFrame(crn_results_a)
        crn_df_b = pd.DataFrame(crn_results_b)
        non_crn_df_a = pd.DataFrame(non_crn_results_a)
        non_crn_df_b = pd.DataFrame(non_crn_results_b)
        
        # Calculate differences
        crn_differences = self._calculate_differences(crn_df_a, crn_df_b, paired=True)
        non_crn_differences = self._calculate_differences(non_crn_df_a, non_crn_df_b, paired=False)
        
        # Calculate variance reduction
        variance_reduction = self._calculate_variance_reduction(
            crn_differences, non_crn_differences)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(
            crn_df_a, crn_df_b, non_crn_df_a, non_crn_df_b)
        
        return CRNComparisonResult(
            config_a=config_a,
            config_b=config_b,
            crn_differences=crn_differences,
            non_crn_differences=non_crn_differences,
            variance_reduction=variance_reduction,
            statistical_tests=statistical_tests
        )
    
    def _extract_metrics(self, results: Dict) -> Dict:
        """Extract key metrics from simulation results"""
        return {
            'population': results['population'],
            'resources': results['resources'],
            'health_index': results['health_index'],
            'distance_covered': results['distance_covered'],
            'years_survived': results['year'],
            'success': 1 if results['status'] == 'Success' else 0
        }
    
    def _run_non_crn_simulations(self, config: Dict, num_runs: int) -> List[Dict]:
        """Run simulations without CRN"""
        results = []
        for i in range(num_runs):
            response = self.session.post(f"{self.base_url}/initialize", 
                                       json={"config": config})
            simulation_id = response.json()['simulation_id']
            sim_results = self.session.post(f"{self.base_url}/simulate/{simulation_id}", 
                                          json={'years': 1000}).json()
            results.append(self._extract_metrics(sim_results[-1]))
            print(f"Completed non-CRN run {i+1}/{num_runs} for configuration")
        return results
    
    def _calculate_differences(self, df_a: pd.DataFrame, df_b: pd.DataFrame, 
                             paired: bool = True) -> pd.DataFrame:
        """Calculate differences between configurations"""
        if paired:
            differences = df_a - df_b
        else:
            # Calculate summary statistics for unpaired comparison
            differences = pd.DataFrame({
                'mean_diff': df_a.mean() - df_b.mean(),
                'std_diff': np.sqrt(df_a.var() + df_b.var())
            }).T
            
        return differences
    
    def _calculate_variance_reduction(self, crn_diff: pd.DataFrame, 
                                    non_crn_diff: pd.DataFrame) -> Dict:
        """Calculate variance reduction ratios for each metric"""
        variance_reduction = {}
        
        for column in crn_diff.columns:
            if column in ['success']:  # Skip binary metrics
                continue
            
            crn_variance = np.var(crn_diff[column])
            non_crn_variance = non_crn_diff.loc['std_diff', column] ** 2
            
            variance_reduction[column] = 1 - (crn_variance / non_crn_variance)
            
        return variance_reduction
    
    def _perform_statistical_tests(self, crn_df_a: pd.DataFrame, crn_df_b: pd.DataFrame,
                                 non_crn_df_a: pd.DataFrame, non_crn_df_b: pd.DataFrame) -> Dict:
        """Perform statistical tests for both CRN and non-CRN results"""
        test_results = {}
        
        # Test metrics
        metrics = [col for col in crn_df_a.columns if col != 'success']
        
        for metric in metrics:
            # CRN (paired t-test)
            crn_ttest = stats.ttest_rel(crn_df_a[metric], crn_df_b[metric])
            
            # Non-CRN (unpaired t-test)
            non_crn_ttest = stats.ttest_ind(non_crn_df_a[metric], non_crn_df_b[metric])
            
            test_results[metric] = {
                'crn': {
                    'statistic': crn_ttest.statistic,
                    'p_value': crn_ttest.pvalue
                },
                'non_crn': {
                    'statistic': non_crn_ttest.statistic,
                    'p_value': non_crn_ttest.pvalue
                }
            }
        
        # Success rate comparison (chi-square test)
        success_a = non_crn_df_a['success'].sum()
        success_b = non_crn_df_b['success'].sum()
        n = len(non_crn_df_a)
        
        contingency = [[success_a, n - success_a],
                      [success_b, n - success_b]]
        
        chi2, p_value = stats.chi2_contingency(contingency)[:2]
        
        test_results['success_rate'] = {
            'chi2': chi2,
            'p_value': p_value
        }
        
        return test_results
    
    def print_comparison_report(self, result: CRNComparisonResult) -> None:
        """Print detailed comparison report"""
        print("\nConfiguration Comparison Report")
        print("=" * 50)
        
        print("\nVariance Reduction with CRN:")
        for metric, reduction in result.variance_reduction.items():
            print(f"{metric}: {reduction*100:.1f}% reduction in variance")
        
        print("\nStatistical Tests:")
        for metric, tests in result.statistical_tests.items():
            if metric != 'success_rate':
                print(f"\n{metric}:")
                print("CRN (Paired t-test):")
                print(f"  t-statistic: {tests['crn']['statistic']:.3f}")
                print(f"  p-value: {tests['crn']['p_value']:.4f}")
                
                print("Non-CRN (Unpaired t-test):")
                print(f"  t-statistic: {tests['non_crn']['statistic']:.3f}")
                print(f"  p-value: {tests['non_crn']['p_value']:.4f}")
            else:
                print("\nSuccess Rate (Chi-square test):")
                print(f"  chi2: {tests['chi2']:.3f}")
                print(f"  p-value: {tests['p_value']:.4f}")

# Example usage:
if __name__ == "__main__":
    # Initialize client
    client = CRNComparisonClient()
    
    # Define two configurations to compare
    config_a = {
        "initial_population": 1000,
        "ship_capacity": 1100,
        "initial_resources": 110000,
        "birth_rate": 9.4,
        "death_rate": 3.7,
        "resource_gen_rate": 99.7,
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    }
    
    config_b = {
        "initial_population": 1000,
        "ship_capacity": 1100,
        "initial_resources": 110000,
        "birth_rate": 9.4,
        "death_rate": 3.7,
        "resource_gen_rate": 102.5,
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    }
    
    # Run comparison
    comparison_result = client.run_crn_comparison(config_a, config_b, num_runs=50)
    
    # Print report
    client.print_comparison_report(comparison_result)