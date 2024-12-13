import requests
import pandas as pd
from typing import Dict, List, Optional
from io import StringIO
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


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
        self.simulation_runs: List[SimulationRun] = []
        
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
        return response.json()

    def simulate(self, years: int = 1) -> List[Dict]:
        response = requests.post(f"{self.base_url}/simulate", json={'years': years})
        response.raise_for_status()
        return response.json()

    def reset(self) -> Dict:
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

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

# Generation Ship Configurations with Balanced Challenges

configurations = {
    "steady_pressure": {
        "initial_population": 1200,     
        "ship_capacity": 1400,          
        "initial_resources": 150000,    
        "birth_rate": 17,              
        "death_rate": 7,               
        "resource_gen_rate": 99,        
        "lightspeed_fraction": 0.006,
        "health_index": 100
    },
    "balanced_crisis": {
        "initial_population": 1300,      
        "ship_capacity": 1600,           
        "initial_resources": 143000,     # 110 per person
        "birth_rate": 18,               
        "death_rate": 6,                
        "resource_gen_rate": 97,        # Right in between our last attempts
        "lightspeed_fraction": 0.007,   
        "health_index": 100
    }
}

"""
Configuration Characteristics:

1. Steady Pressure:
   - Moderate population growth (1% per year)
   - Slight resource deficit (99% generation)
   - More resources per person
   - Expected success rate: Variable
   - Tests gradual adaptation

2. Balanced Crisis:
   - Starts at consumption limit
   - Higher growth pressure (1.2% per year)
   - Immediate resource pressure
   - Expected success rate: Variable
   - Tests crisis management

Both configurations should produce interesting outcomes by:
- Making tiny populations non-viable
- Creating multiple competing pressures
- Requiring population management
- Testing different survival strategies
"""

# Example usage with the enhanced client
if __name__ == "__main__":
    
    client = EnhancedGenerationShipClient()
    
    # Run multiple simulations for each configuration
    for config_name, config in configurations.items():
        print(f"\nRunning simulations for {config_name}")
        try:
            client.run_multiple_simulations(config, num_runs=100, max_years=1000)
            detailed_results, summary_stats = client.generate_summary_report(f"{config_name}_results.xlsx")
            print("\nSimulation Summary Statistics:")
            print(summary_stats.to_string(index=False))
        except Exception as e:
            print(f"Error running {config_name}: {str(e)}")