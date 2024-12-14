import numpy as np
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from scipy.stats import skewnorm
import csv
from io import StringIO
from uuid import uuid4
from datetime import datetime, timedelta
import threading

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Constants
DISTANCE_TO_PROXIMA_CENTAURI = 40140000000000
SPEED_OF_LIGHT_KM_PER_HR = 1079251200
NORMAL_CONSUMPTION = 100
CRITICAL_CONSUMPTION_THRESHOLD = 90

class SimulationManager:
    def __init__(self):
        self.simulations = {}
        self.lock = threading.Lock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_simulations, daemon=True)
        self._cleanup_thread.start()
    
    def create_simulation(self, config, crn_seed=None):
        """Create a new simulation instance with optional CRN"""
        simulation_id = str(uuid4())
        with self.lock:
            self.simulations[simulation_id] = {
                'instance': GenerationShip(config, crn_seed),
                'last_accessed': datetime.now()
            }
        return simulation_id
    
    def get_simulation(self, simulation_id):
        with self.lock:
            if simulation_id in self.simulations:
                self.simulations[simulation_id]['last_accessed'] = datetime.now()
                return self.simulations[simulation_id]['instance']
        return None
    
    def remove_simulation(self, simulation_id):
        with self.lock:
            if simulation_id in self.simulations:
                del self.simulations[simulation_id]
    
    def _cleanup_expired_simulations(self):
        while True:
            current_time = datetime.now()
            with self.lock:
                expired = [
                    sim_id for sim_id, sim_data in self.simulations.items()
                    if (current_time - sim_data['last_accessed']) > timedelta(minutes=30)
                ]
                for sim_id in expired:
                    del self.simulations[sim_id]
            threading.Event().wait(300)

simulation_manager = SimulationManager()

class GenerationShip:
    def __init__(self, data, crn_seed=None):
        """
        Initialize generation ship simulation with optional CRN
        
        Args:
            data: Configuration dictionary
            crn_seed: If provided, enables CRN mode with this seed as base
        """
        # First validate the data
        self.validate_data(data)
        
        # Store original config for resets
        self.config = data.copy()
        self.crn_seed = crn_seed
        
        # Initialize all state variables
        self.year = 0
        self.population = data['initial_population']
        self.resources = data['initial_resources']
        self.ship_capacity = data['ship_capacity']
        self.initial_birth_rate = data['birth_rate']
        self.initial_death_rate = data['death_rate']
        self.birth_rate = self.initial_birth_rate  # Set initial birth rate
        self.death_rate = self.initial_death_rate  # Set initial death rate
        self.initial_resource_gen_rate = data['resource_gen_rate']
        self.resource_gen_rate = self.initial_resource_gen_rate
        self.speed = data['lightspeed_fraction'] * SPEED_OF_LIGHT_KM_PER_HR
        self.total_distance = DISTANCE_TO_PROXIMA_CENTAURI
        self.total_years = (self.total_distance / (self.speed * 24 * 365))
        self.distance_covered = 0
        self.initial_health_index = data['health_index']
        self.health_index = self.initial_health_index
        self.status = "Running"
        
        # Initialize event flags
        self.diseaseOutbreakEvent = 0
        self.overCrowdingEvent = 0
        self.criticalRationingEvent = 0
        self.normalRationingEvent = 0
        
        # Initialize history
        self.simulation_history = []
        
        # Initialize RNG last
        self.initialize_rng(crn_seed)
        
    def initialize_state(self, data):
        """Initialize simulation state variables"""
        self.year = 0
        self.population = data['initial_population']
        self.resources = data['initial_resources']
        self.ship_capacity = data['ship_capacity']
        self.initial_birth_rate = data['birth_rate']
        self.initial_death_rate = data['death_rate']
        self.initial_resource_gen_rate = data['resource_gen_rate']
        self.speed = data['lightspeed_fraction'] * SPEED_OF_LIGHT_KM_PER_HR
        self.total_distance = DISTANCE_TO_PROXIMA_CENTAURI
        self.total_years = (self.total_distance / (self.speed * 24 * 365))
        self.distance_covered = 0
        self.initial_health_index = data['health_index']
        self.status = "Running"
        self.simulation_history = []
        self.reset_events()

    def initialize_rng(self, crn_seed):
        """Initialize random number generators"""
        if crn_seed is not None:
            # Use CRN mode with different seeds for each RNG
            self.disease_rng = np.random.RandomState(crn_seed)
            self.births_rng = np.random.RandomState(crn_seed + 1)
            self.deaths_rng = np.random.RandomState(crn_seed + 2)
            self.resource_prod_rng = np.random.RandomState(crn_seed + 3)
            self.resource_cons_rng = np.random.RandomState(crn_seed + 4)
            self.using_crn = True
        else:
            # Use system RNG for all random numbers
            self.disease_rng = np.random
            self.births_rng = np.random
            self.deaths_rng = np.random
            self.resource_prod_rng = np.random
            self.resource_cons_rng = np.random
            self.using_crn = False

    def reset_events(self):
        """Reset event flags"""
        self.diseaseOutbreakEvent = 0
        self.overCrowdingEvent = 0
        self.criticalRationingEvent = 0
        self.normalRationingEvent = 0
        
    def reset(self, new_crn_seed=None):
        """Reset simulation to initial state with optional new CRN seed"""
        if new_crn_seed is not None:
            self.crn_seed = new_crn_seed
        
        # Reset all state variables to initial values
        self.year = 0
        self.population = self.config['initial_population']
        self.resources = self.config['initial_resources']
        self.birth_rate = self.initial_birth_rate
        self.death_rate = self.initial_death_rate
        self.resource_gen_rate = self.initial_resource_gen_rate
        self.distance_covered = 0
        self.health_index = self.initial_health_index
        self.status = "Running"
        
        # Clear events
        self.diseaseOutbreakEvent = 0
        self.overCrowdingEvent = 0
        self.criticalRationingEvent = 0
        self.normalRationingEvent = 0
        
        # Clear history
        self.simulation_history = []
        
        # Reinitialize RNG if needed
        if new_crn_seed is not None:
            self.initialize_rng(new_crn_seed)
        
    @staticmethod
    def validate_data(data):
        required_keys = [
            'initial_population', 'ship_capacity', 'initial_resources',
            'birth_rate', 'death_rate', 'resource_gen_rate',
            'lightspeed_fraction', 'health_index'
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required parameter: {key}")
            if not isinstance(data[key], (int, float)) or data[key] <= 0:
                raise ValueError(f"Invalid value for {key}: must be a positive number")

    def simulate_year(self):
        """Simulate one year with either CRN or standard random numbers"""
        if self.status in ["Failed", "Success"]:
            return []

        self.reset_events()
        current_year_log = []
        
        # Update resource generation rate and health
        self.resource_gen_rate = self.initial_resource_gen_rate
        self.health_index = max(0, min(100, self.initial_health_index))

        # Population density effects
        self._handle_overcrowding(current_year_log)
        
        # Disease outbreaks
        self._handle_disease_outbreaks(current_year_log)
        
        # Population changes
        if not self._handle_population_changes(current_year_log):
            return current_year_log
        
        # Resource management
        if not self._handle_resources(current_year_log):
            return current_year_log
            
        # Distance and mission status
        self._update_mission_status(current_year_log)
        
        # Update history and return
        self._append_history()
        return current_year_log

    def _handle_overcrowding(self, logs):
        """Handle overcrowding effects"""
        if self.population > 0.9 * self.ship_capacity:
            overcrowding_penalty = (self.population / self.ship_capacity - 0.8) * 10
            self.health_index *= 0.90
            self.resource_gen_rate *= 0.8
            self.overCrowdingEvent = 1
            logs.append("Overcrowding detected! Health and production efficiency reduced.")

    def _handle_disease_outbreaks(self, logs):
        """Handle disease outbreak events"""
        if self.disease_rng.random() < 0.05:
            disease_penalty = self.disease_rng.randint(1, 5) / 10
            self.health_index *= (1 - disease_penalty)
            self.diseaseOutbreakEvent = 1
            logs.append(f"Disease outbreak! Health index dropped by {disease_penalty*100:.1f}%")

    def _handle_population_changes(self, logs):
        """Handle births and deaths"""
        health_factor = self.health_index / 100
        self.birth_rate = max(0, self.initial_birth_rate * health_factor)
        self.death_rate = max(0, self.initial_death_rate * (2 - health_factor))
        
        if self.population > 0:
            births = self.births_rng.poisson(self.birth_rate * self.population / 1000)
            deaths = self.deaths_rng.poisson(self.death_rate * self.population / 1000)
            self.population += births - deaths
            logs.append(f"Births: {births}, Deaths: {deaths}")
            
            # Handle capacity limits
            if self.population > self.ship_capacity:
                excess = self.population - self.ship_capacity
                self.population = self.ship_capacity
                self.overCrowdingEvent = 1
                logs.append(f"Population exceeded capacity! {excess} people lost.")
        
        self.population = max(0, self.population)
        
        if self.population <= 0:
            self.status = "Failed"
            logs.append("Mission failed! Population reached zero.")
            return False
            
        return True

    def _handle_resources(self, logs):
        """Handle resource production and consumption"""
        resources_per_person = self.resources / self.population if self.population > 0 else 0
        
        # Production
        if self.population > 0:
            base_production = self.population * self.resource_gen_rate
            resource_production = max(0, skewnorm.rvs(-1,
                loc=base_production * (self.health_index / 100),
                scale=base_production * 0.1,
                random_state=self.resource_prod_rng))
        else:
            resource_production = 0
            
        # Consumption
        if resources_per_person < CRITICAL_CONSUMPTION_THRESHOLD:
            self._handle_critical_resources(logs)
        elif resources_per_person < NORMAL_CONSUMPTION:
            self._handle_low_resources(logs)
        else:
            self._handle_normal_resources(logs)
            
        resource_consumption = self._calculate_consumption()
        
        # Update resources
        self.resources = max(0, self.resources + resource_production - resource_consumption)
        logs.append(f"Resources: Produced: {resource_production:.2f}, "
                   f"Consumed: {resource_consumption:.2f}, "
                   f"Remaining: {self.resources:.2f}")
        
        if self.resources <= 0:
            self.status = "Failed"
            logs.append("Mission failed! Resources depleted.")
            return False
            
        return True

    def _handle_critical_resources(self, logs):
        """Handle critical resource shortage"""
        sudden_loss = max(1, int(0.1 * self.population))
        self.population -= sudden_loss
        self.resource_gen_rate *= 0.5
        self.criticalRationingEvent = 1
        logs.append(f"Critical resource shortage! Lost {sudden_loss} population.")

    def _handle_low_resources(self, logs):
        """Handle low resource situation"""
        self.health_index *= 0.9
        self.resource_gen_rate *= 0.9
        self.normalRationingEvent = 1
        logs.append("Resource rationing in effect.")

    def _handle_normal_resources(self, logs):
        """Handle normal resource situation"""
        self.health_index *= 1.1
        logs.append("Resource levels normal.")

    def _calculate_consumption(self):
        """Calculate resource consumption based on current state"""
        if self.criticalRationingEvent:
            base_consumption = CRITICAL_CONSUMPTION_THRESHOLD
        elif self.normalRationingEvent:
            base_consumption = (NORMAL_CONSUMPTION + CRITICAL_CONSUMPTION_THRESHOLD) / 2
        else:
            base_consumption = NORMAL_CONSUMPTION
            
        consumption = self.resource_cons_rng.normal(
            base_consumption, 10, size=int(self.population)
        ).sum()
        
        return max(0, min(consumption, self.resources))

    def _update_mission_status(self, logs):
        """Update mission progress and status"""
        self.distance_covered += self.speed * 24 * 365
        self.year += 1
        self.health_index *= 0.98  # Age penalty
        
        if self.distance_covered >= DISTANCE_TO_PROXIMA_CENTAURI:
            self.status = "Success"
            logs.append("Successfully reached Proxima Centauri!")
        else:
            logs.append(f"Distance covered: {self.distance_covered:.2f} km")

    def _append_history(self):
        """Record current state in simulation history"""
        self.simulation_history.append({
            "year": self.year,
            "population": self.population,
            "resources": self.resources,
            "distance_covered": self.distance_covered,
            "health_index": self.health_index,
            "birth_rate": self.birth_rate,
            "death_rate": self.death_rate,
            "resource_gen_rate": self.resource_gen_rate,
            "status": self.status,
            "diseaseOutbreakEvent": self.diseaseOutbreakEvent,
            "overCrowdingEvent": self.overCrowdingEvent,
            "criticalRationingEvent": self.criticalRationingEvent,
            "normalRationingEvent": self.normalRationingEvent
        })

    def get_status(self, logs=None):
        """Get current simulation status"""
        return {
            "year": self.year,
            "totalYears": self.total_years,
            "population": max(0, self.population),
            "resources": max(0, self.resources),
            "distance_covered": self.distance_covered,
            "ship_capacity": self.ship_capacity,
            "totalDistance": self.total_distance,
            "birth_rate": self.birth_rate,
            "death_rate": self.death_rate,
            "health_index": self.health_index,
            "speed": self.speed,
            "resource_gen_rate": self.resource_gen_rate,
            "status": self.status,
            "using_crn": self.using_crn,
            "crn_seed": self.crn_seed if self.using_crn else None,
            "log": logs if logs else [],
            "diseaseOutbreakEvent": self.diseaseOutbreakEvent,
            "overCrowdingEvent": self.overCrowdingEvent,
            "criticalRationingEvent": self.criticalRationingEvent,
            "normalRationingEvent": self.normalRationingEvent
        }

    def get_csv(self):
        """Export simulation history as CSV"""
        output = StringIO()
        writer = csv.writer(output)
        
        headers = ["Year", "Population", "Resources", "Distance Covered (km)", 
                  "Health Index", "Birth Rate", "Death Rate", 
                  "Resource Generation Rate", "Status"]
        writer.writerow(headers)
        
        for record in self.simulation_history:
            writer.writerow([
                record["year"],
                record["population"],
                f"{record['resources']:.2f}",
                f"{record['distance_covered']:.2f}",
                f"{record['health_index']:.2f}",
                f"{record['birth_rate']:.4f}",
                f"{record['death_rate']:.4f}",
                f"{record['resource_gen_rate']:.2f}",
                record["status"]
            ])
            
        return output.getvalue()

# Flask Routes
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize a new simulation with optional CRN"""
    try:
        logger.debug("Received initialize request")
        data = request.json
        logger.debug(f"Received data: {data}")
        
        # Support both new format {"config": {...}, "crn_seed": ...} 
        # and old format where config is the root object
        if isinstance(data, dict) and "config" in data:
            config = data["config"]
            crn_seed = data.get("crn_seed")
        else:
            config = data
            crn_seed = None
            
        logger.debug(f"Processed config: {config}")
        logger.debug(f"CRN seed: {crn_seed}")

        # Validate config before creating simulation
        if not isinstance(config, dict):
            raise ValueError(f"Invalid config format: {type(config)}")

        # Create new simulation instance
        simulation_id = simulation_manager.create_simulation(config, crn_seed)
        logger.debug(f"Created simulation with ID: {simulation_id}")
        
        # Get initial status
        simulation = simulation_manager.get_simulation(simulation_id)
        initial_status = simulation.get_status()
        
        response = {
            "message": "Simulation initialized!",
            "simulation_id": simulation_id,
            "using_crn": crn_seed is not None,
            "crn_seed": crn_seed,
            "initial_status": initial_status
        }
        logger.debug(f"Sending response: {response}")
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"ValueError in initialize: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in initialize: {str(e)}", exc_info=True)
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/simulate/<simulation_id>', methods=['POST'])
def simulate(simulation_id):
    """Run simulation for specified number of years"""
    try:
        ship = simulation_manager.get_simulation(simulation_id)
        if not ship:
            return jsonify({"error": "Invalid or expired simulation ID"}), 404

        # Get number of years to simulate
        years = request.json.get('years', 1)
        if not isinstance(years, int) or years < 1:
            return jsonify({"error": "Years must be a positive integer"}), 400

        # Run simulation
        results = []
        for _ in range(years):
            current_logs = ship.simulate_year()
            status = ship.get_status(logs=current_logs)
            results.append(status)

            if ship.status in ["Failed", "Success"]:
                break

        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Simulation error: {str(e)}"}), 500

@app.route('/reset/<simulation_id>', methods=['POST'])
def reset(simulation_id):
    """Reset simulation to initial state"""
    try:
        ship = simulation_manager.get_simulation(simulation_id)
        if not ship:
            return jsonify({"error": "Invalid or expired simulation ID"}), 404
        
        # Get optional new CRN seed
        data = request.json or {}
        new_crn_seed = data.get('crn_seed')
        
        # Reset simulation
        ship.reset(new_crn_seed)
        
        return jsonify({
            "message": "Simulation reset successfully",
            "using_crn": ship.using_crn,
            "crn_seed": ship.crn_seed if ship.using_crn else None
        })
        
    except Exception as e:
        return jsonify({"error": f"Reset error: {str(e)}"}), 500

@app.route('/export-csv/<simulation_id>', methods=['GET'])
def export_csv(simulation_id):
    """Export simulation history as CSV"""
    try:
        ship = simulation_manager.get_simulation(simulation_id)
        if not ship:
            return jsonify({"error": "Invalid or expired simulation ID"}), 404
        
        if not ship.simulation_history:
            return jsonify({"error": "No simulation data available"}), 400
        
        csv_data = ship.get_csv()
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=generation_ship_simulation.csv'
        
        return response
        
    except Exception as e:
        return jsonify({"error": f"Export error: {str(e)}"}), 500

@app.route('/status/<simulation_id>', methods=['GET'])
def get_status(simulation_id):
    """Get current simulation status"""
    try:
        ship = simulation_manager.get_simulation(simulation_id)
        if not ship:
            return jsonify({"error": "Invalid or expired simulation ID"}), 404
            
        return jsonify(ship.get_status())
        
    except Exception as e:
        return jsonify({"error": f"Status error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)