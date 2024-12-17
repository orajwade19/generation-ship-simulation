import requests
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
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
    resource_history: List[float] = None
    average_resources: float = None 


class ParallelGenerationShipClient:
    def __init__(self, base_url: str = 'http://localhost:5001', max_workers: int = 10):
        self.base_url = base_url.rstrip('/')
        self.max_workers = max_workers
        self.session = requests.Session()
        self.simulation_runs = []

    def _run_single_simulation(self, run_id: int, config: Dict, max_years: int = 1000, crn_seed: Optional[int] = None) -> SimulationRun:
        """Run a single simulation with CRN if seed is provided"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                init_payload = {"config": config}
                if crn_seed is not None:
                    init_payload["crn_seed"] = crn_seed

                # Initialize simulation
                response = self.session.post(f"{self.base_url}/initialize", json=init_payload)
                response.raise_for_status()
                simulation_id = response.json()['simulation_id']

                # Run full simulation at once
                response = self.session.post(
                    f"{self.base_url}/simulate/{simulation_id}", 
                    json={'years': max_years}
                )
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    raise Exception("Simulation failed with no results")
                    
                last_year = results[-1]
                resource_history = [year['resources'] for year in results]
                average_resources = float(np.mean(resource_history)) if resource_history else 0.0

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
                    normal_rationing_events=normal_rationing_events,
                    resource_history=resource_history,
                    average_resources=average_resources
                )
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed to complete simulation {run_id} after {max_retries} attempts: {str(e)}")
                    raise
                time.sleep(retry_delay * (2 ** attempt))
                continue

    def run_parallel_simulations(self, config: Dict, num_runs: int = 10, max_years: int = 1000, crn_seed_base: Optional[int] = None) -> List[SimulationRun]:
        """Run multiple simulations in parallel, optionally using CRN seeds."""
        self.simulation_runs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(num_runs):
                seed = crn_seed_base + i if crn_seed_base is not None else None
                futures.append(executor.submit(self._run_single_simulation, i, config, max_years, seed))
                
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                self.simulation_runs.append(result)
        
        return self.simulation_runs

    def analyze_results(self) -> pd.DataFrame:
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
                'Average Resources': run.average_resources,
                'Final Health': run.final_health,
                'Distance Covered (km)': run.distance_covered,
                'Disease Outbreaks': run.disease_outbreaks,
                'Overcrowding Events': run.overcrowding_events,
                'Critical Rationing Events': run.critical_rationing_events,
                'Normal Rationing Events': run.normal_rationing_events
            }
            for key, value in run.config.items():
                run_dict[f'Config_{key}'] = value
            runs_data.append(run_dict)
            
        return pd.DataFrame(runs_data)

    def get_summary_statistics(self) -> pd.DataFrame:
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
    

class CRNGridSearchClient(ParallelGenerationShipClient):
    def run_crn_grid_search(self, configs: List[Dict], num_runs: int = 50, max_years: int = 1000, base_seed: int = 42) -> pd.DataFrame:
        """
        Perform a grid search over the provided list of configurations using CRN for reproducibility.
        
        Args:
            configs: List of configuration dictionaries to test
            num_runs: Number of runs per configuration
            max_years: Max simulation years
            base_seed: Base seed for CRN
            
        Returns:
            A DataFrame with results for all configurations
        """
        all_results = []
        total_configs = len(configs)
        
        for i, config in enumerate(configs, start=1):
            print(f"\nRunning CRN simulations for configuration {i}/{total_configs}...")
            self.run_parallel_simulations(config, num_runs=num_runs, max_years=max_years, crn_seed_base=base_seed)
            df = self.analyze_results()
            # Add a unique ID or tuple of parameters to identify the configuration
            df['ship_capacity'] = config['ship_capacity']
            df['initial_resources'] = config['initial_resources']
            df['resource_gen_rate'] = config['resource_gen_rate']
            all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    # Define parameter ranges for the grid search
# Capacities: steps of 25 instead of 50
    capacities = [1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250]

# Resources: steps of 5000 instead of 10000
    resources = [90000,95000,100000, 105000, 110000, 115000, 120000, 125000, 130000]


# Resource generation rates: steps of 1.0 instead of 2.5
    gen_rates = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

    # Fixed parameters
    initial_population = 1000
    birth_rate = 9.4
    death_rate = 3.7
    lightspeed_fraction = 0.0059
    health_index = 100

    # Generate all configurations
    config_list = []
    for c, r, g in product(capacities, resources, gen_rates):
        config = {
            "initial_population": initial_population,
            "ship_capacity": c,
            "initial_resources": r,
            "birth_rate": birth_rate,
            "death_rate": death_rate,
            "resource_gen_rate": g,
            "lightspeed_fraction": lightspeed_fraction,
            "health_index": health_index
        }
        config_list.append(config)
    
    # Initialize the client
    client = CRNGridSearchClient()

    # Run the grid search with CRN
    results_df = client.run_crn_grid_search(config_list, num_runs=100, max_years=1000, base_seed=42)
    reduced_df = results_df[['ship_capacity', 'initial_resources', 'resource_gen_rate', 'Final Population', 'Final Status','Average Resources']]

    summary_df = reduced_df.groupby(['ship_capacity', 'initial_resources', 'resource_gen_rate']).agg(
    mean_final_population=('Final Population', 'mean'),
    var_final_population=('Final Population', 'var'),
    mean_average_resources=('Average Resources', 'mean'),
    var_average_resources=('Average Resources', 'var'),
    success_rate=('Final Status', lambda x: (x == 'Success').mean() * 100)
    ).reset_index()
    print("\nGrid Search Results:")
    print(results_df.head())
    results_df.to_csv("grid_search_results.csv", index=False)
    summary_df.to_csv("summary_results.csv", index=False)

