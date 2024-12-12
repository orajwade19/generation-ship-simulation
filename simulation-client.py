import requests
import pandas as pd
from typing import Dict, List, Optional
from io import StringIO

class GenerationShipClient:
    def __init__(self, base_url: str = 'http://localhost:5001'):
        self.base_url = base_url.rstrip('/')

    def initialize(self, config: Dict) -> Dict:
        """
        Initialize the generation ship simulation with configuration parameters.
        
        Args:
            config: Dictionary containing:
                - initial_population: int
                - ship_capacity: int
                - initial_resources: float
                - birth_rate: float
                - death_rate: float
                - resource_gen_rate: float
                - lightspeed_fraction: float
                - health_index: float
        
        Returns:
            Dict containing initialization status
        """
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
        """
        Run the simulation for specified number of years.
        
        Args:
            years: Number of years to simulate
        
        Returns:
            List of dictionaries containing simulation results for each year
        """
        response = requests.post(f"{self.base_url}/simulate", json={'years': years})
        response.raise_for_status()
        return response.json()

    def reset(self) -> Dict:
        """
        Reset the simulation to initial state.
        
        Returns:
            Dict containing reset status
        """
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

    def export_csv(self) -> str:
        """
        Export simulation results as CSV string.
        
        Returns:
            String containing CSV data
        """
        response = requests.get(f"{self.base_url}/export-csv")
        response.raise_for_status()
        return response.text

    def export_dataframe(self) -> pd.DataFrame:
        """
        Export simulation results as pandas DataFrame.
        
        Returns:
            pandas DataFrame containing simulation results
        """
        csv_data = self.export_csv()
        return pd.read_csv(StringIO(csv_data))

    def save_csv(self, filename: str = 'generation_ship_simulation.csv'):
        """
        Save simulation results to a CSV file.
        
        Args:
            filename: Name of the file to save
        """
        csv_data = self.export_csv()
        with open(filename, 'w') as f:
            f.write(csv_data)

# Example usage
if __name__ == "__main__":
    client = GenerationShipClient()

    # Initialize simulation
    config = {
        'initial_population': 1000,
        'ship_capacity': 2000,
        'initial_resources': 100000,
        'birth_rate': 15,  # per 1000 people
        'death_rate': 8,   # per 1000 people
        'resource_gen_rate': 150,
        'lightspeed_fraction': 0.15,
        'health_index': 100
    }

    try:
        # Initialize and run simulation
        client.initialize(config)
        results = client.simulate(years=10)
        
        # Export results in different formats
        client.save_csv('simulation_results.csv')  # Save to CSV file
        df = client.export_dataframe()  # Get pandas DataFrame
        print(df.head())
        
        # Reset simulation if needed
        client.reset()
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
    except ValueError as e:
        print(f"Validation Error: {e}")