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

# Constants remain the same
DISTANCE_TO_PROXIMA_CENTAURI = 40140000000000
SPEED_OF_LIGHT_KM_PER_HR = 1079251200
NORMAL_CONSUMPTION = 100
CRITICAL_CONSUMPTION_THRESHOLD = 90

# Simulation manager to handle multiple instances
class SimulationManager:
    def __init__(self):
        self.simulations = {}
        self.lock = threading.Lock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_simulations, daemon=True)
        self._cleanup_thread.start()
    
    def create_simulation(self, config):
        """Create a new simulation instance and return its ID"""
        simulation_id = str(uuid4())
        with self.lock:
            self.simulations[simulation_id] = {
                'instance': GenerationShip(config),
                'last_accessed': datetime.now()
            }
        return simulation_id
    
    def get_simulation(self, simulation_id):
        """Get a simulation instance by ID and update its last accessed time"""
        with self.lock:
            if simulation_id in self.simulations:
                self.simulations[simulation_id]['last_accessed'] = datetime.now()
                return self.simulations[simulation_id]['instance']
        return None
    
    def remove_simulation(self, simulation_id):
        """Remove a simulation instance"""
        with self.lock:
            if simulation_id in self.simulations:
                del self.simulations[simulation_id]
    
    def _cleanup_expired_simulations(self):
        """Periodically remove simulations that haven't been accessed in 30 minutes"""
        while True:
            current_time = datetime.now()
            with self.lock:
                expired = [
                    sim_id for sim_id, sim_data in self.simulations.items()
                    if (current_time - sim_data['last_accessed']) > timedelta(minutes=30)
                ]
                for sim_id in expired:
                    del self.simulations[sim_id]
            # Sleep for 5 minutes before next cleanup
            threading.Event().wait(300)

# Global simulation manager
simulation_manager = SimulationManager()

# GenerationShip class remains mostly the same, just remove global state
class GenerationShip:
    def __init__(self, data):
        # Existing initialization code remains the same
        self.validate_data(data)
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
        self.diseaseOutbreakEvent = 0
        self.overCrowdingEvent = 0
        self.criticalRationingEvent = 0
        self.normalRationingEvent = 0
        self.simulation_history = []

    @staticmethod
    def validate_data(data):
        required_keys = ['initial_population', 'ship_capacity', 'initial_resources', 'birth_rate', 'death_rate',
                         'resource_gen_rate', 'lightspeed_fraction']
        for key in required_keys:
            if key not in data or data[key] <= 0:
                raise ValueError(f"Invalid or missing value for {key}")

    def simulate_year(self):
        self.diseaseOutbreakEvent = 0
        self.overCrowdingEvent = 0
        self.criticalRationingEvent = 0
        self.normalRationingEvent = 0
        self.resource_gen_rate = self.initial_resource_gen_rate
        self.health_index = self.initial_health_index
        if self.status in ["Failed", "Success"]:
            return []

        current_year_log = []

        # Calculate Health Index
        self.health_index = max(0, min(100, self.health_index))  # Clamp health between 0 and 100

        # Check for overpopulation
        if self.population > 0.9 * self.ship_capacity:
            overcrowding_penalty = (self.population / self.ship_capacity - 0.8) * 10  # Penalty based on excess
            self.health_index *= 0.90  # Reduce health index due to overcrowding
            self.resource_gen_rate *= 0.8  # Decrease resource production efficiency
            self.overCrowdingEvent = 1
            current_year_log.append(f"Overcrowding detected! Health index reduced by 5%. Production efficiency reduced.")

        # Random disease outbreaks
        if np.random.random() < 0.05:  # 5% chance of disease outbreak
            disease_penalty = np.random.randint(1, 5) / 10
            self.health_index *= (1 - disease_penalty)
            self.diseaseOutbreakEvent = 1
            current_year_log.append(f"Disease outbreak! Health index dropped by {disease_penalty}.")

        # Adjust birth and death rates based on health index
        health_factor = self.health_index / 100
        self.birth_rate = max(0, self.initial_birth_rate * health_factor)  # Ensure non-negative birth rate
        self.death_rate = max(0, self.initial_death_rate * (2 - health_factor))  # Ensure non-negative death rate

        current_year_log.append(f"Birth Rate: {self.birth_rate:.2f}. Death Rate: {self.death_rate:.2f}")

        # Calculate births and deaths
        if self.population > 0:
            births = np.random.poisson(self.birth_rate * self.population / 1000)
            deaths = np.random.poisson(self.death_rate * self.population / 1000)
        else:
            births = 0
            deaths = 0

        self.population += births - deaths

        # Prevent population exceeding ship capacity
        if self.population > self.ship_capacity:
            excess_population = self.population - self.ship_capacity
            self.population = self.ship_capacity
            self.overCrowdingEvent = 1
            current_year_log.append(f"Population exceeded ship capacity! {excess_population} people lost.")

        self.population = max(0, self.population)

        if self.population <= 0:
            resource_consumption = 0
            self.status = "Failed"
            current_year_log.append("Mission failed! Population reached zero.")
            self.simulation_history.append({
                "year": self.year,
                "population": 0,
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
            return current_year_log


        # Log population changes
        current_year_log.append(f"Births: {births}, Deaths: {deaths}, Population: {self.population}.")

        # Calculate Base Resource Production
        resources_per_person = self.resources / self.population if self.population > 0 else 0

        # Resource Production
        if self.population > 0:
            base_production_rate = self.population * self.resource_gen_rate # Scale by population
            skewness = -1  # Left-tailed skew for realistic variability
            resource_production = max(0, skewnorm.rvs(skewness, loc=base_production_rate * health_factor,
                                                      scale=base_production_rate * 0.1))
        else:
            resource_production = 0

        # Resource Consumption
        if resources_per_person < CRITICAL_CONSUMPTION_THRESHOLD:
            # Critical resource shortage
            sudden_loss = max(1, int(0.1 * self.population))  # Lose 10% of the population
            self.population -= sudden_loss
            resource_consumption = np.random.normal(CRITICAL_CONSUMPTION_THRESHOLD, 10, size=int(self.population)).sum()  # Reduced consumption
            resource_consumption = max(0, min(resource_consumption, self.resources))
            current_year_log.append(f"Critical resource shortage. Population reduced by prioritization ({sudden_loss} lost).")
            self.resource_gen_rate *= 0.5  # Reduced productivity
            self.criticalRationingEvent = 1
        elif resources_per_person < NORMAL_CONSUMPTION:
            # Low resources, rationing
            self.health_index *= 0.9  # Penalty for insufficient resources
            resource_consumption = np.random.normal(CRITICAL_CONSUMPTION_THRESHOLD, 10, size=int(self.population)).sum()  # Rationed consumption
            resource_consumption = max(0, min(resource_consumption, self.resources))
            self.resource_gen_rate *= 0.9
            current_year_log.append("Rationing activated due to low resources.")
            self.normalRationingEvent = 1
        else:
            # Normal resources
            self.health_index *= 1.1  # Bonus for surplus resources
            self.resource_gen_rate *= 1  # Increase productivity
            resource_consumption = np.random.normal(NORMAL_CONSUMPTION, 10, size=int(self.population)).sum()  # Normal consumption
            resource_consumption = max(0, min(resource_consumption, self.resources))

        # Update Resources
        self.resources = max(0, self.resources + resource_production - resource_consumption)

        # Log resource production and consumption
        current_year_log.append(
            f"Resources: Produced: {resource_production:.2f}, Consumed: {resource_consumption:.2f}, Remaining: {self.resources:.2f}"
        )

        # Check for failure or success
        if self.resources <= 0 or self.population <= 0:
            self.status = "Failed"
            current_year_log.append("Mission failed! Resources or population depleted.")
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
                "criticalRationingEvent":self.criticalRationingEvent,
                "normalRationingEvent": self.normalRationingEvent
            })
            return current_year_log

        self.distance_covered += self.speed * 24 * 365  # Distance covered in a year
        if self.distance_covered >= DISTANCE_TO_PROXIMA_CENTAURI:
            self.status = "Success"
            current_year_log.append("Successfully reached Proxima Centauri.")
        else:
            current_year_log.append(f"Distance covered: {self.distance_covered:.2f} km.")

        self.year += 1  # Increment year
        # Age-related health penalty (simple decay model)
        self.health_index *= 0.98
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
            "criticalRationingEvent":self.criticalRationingEvent,
            "normalRationingEvent": self.normalRationingEvent
            })
        return current_year_log

    def reset(self):
        self.year = 0
        self.population = 0
        self.resources = 0
        self.distance_covered = 0
        self.status = ""
        self.health_index = 100
        self.diseaseOutbreakEvent = 0
        self.overCrowdingEvent = 0
        self.criticalRationingEvent = 0
        self.normalRationingEvent = 0

    def get_status(self, logs=None):
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
            "log": logs if logs else [],
            "diseaseOutbreakEvent": self.diseaseOutbreakEvent,
            "overCrowdingEvent": self.overCrowdingEvent,
            "criticalRationingEvent":self.criticalRationingEvent,
            "normalRationingEvent": self.normalRationingEvent
        }


    def get_csv(self):
        """Generate CSV data from simulation history"""
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        headers = ["Year", "Population", "Resources", "Distance Covered (km)", 
                  "Health Index", "Birth Rate", "Death Rate", 
                  "Resource Generation Rate", "Status"]
        writer.writerow(headers)

        # Write data rows
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
# Updated Flask routes to use SimulationManager
@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        data = request.json
        simulation_id = simulation_manager.create_simulation(data)
        return jsonify({
            "message": "Simulation initialized!",
            "simulation_id": simulation_id
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/simulate/<simulation_id>', methods=['POST'])
def simulate(simulation_id):
    ship = simulation_manager.get_simulation(simulation_id)
    if not ship:
        return jsonify({"error": "Invalid or expired simulation ID"}), 404

    years = request.json.get('years', 1)
    results = []

    for _ in range(years):
        current_logs = ship.simulate_year()
        status = ship.get_status(logs=current_logs)
        results.append(status)

        if ship.status in ["Failed", "Success"]:
            break

    return jsonify(results)

@app.route('/reset/<simulation_id>', methods=['POST'])
def reset(simulation_id):
    ship = simulation_manager.get_simulation(simulation_id)
    if not ship:
        return jsonify({"error": "Invalid or expired simulation ID"}), 404
    
    ship.reset()
    return jsonify({"message": "Simulation reset successfully."})

@app.route('/export-csv/<simulation_id>', methods=['GET'])
def export_csv(simulation_id):
    ship = simulation_manager.get_simulation(simulation_id)
    if not ship:
        return jsonify({"error": "Invalid or expired simulation ID"}), 404
    
    if not ship.simulation_history:
        return jsonify({"error": "No simulation data available."}), 400
    
    csv_data = ship.get_csv()
    response = make_response(csv_data)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=generation_ship_simulation.csv'
    
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)