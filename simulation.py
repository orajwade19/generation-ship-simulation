import numpy as np
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from scipy.stats import skewnorm
import csv
from io import StringIO

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Constants
DISTANCE_TO_PROXIMA_CENTAURI = 40140000000000  # In kilometers (4.24 light-years)
SPEED_OF_LIGHT_KM_PER_HR = 1079251200
NORMAL_CONSUMPTION = 100  # Average resource consumption per person
CRITICAL_CONSUMPTION_THRESHOLD = 90  # 50% of normal consumption

# Simulation class
class GenerationShip:
    def __init__(self, data):
        self.validate_data(data)
        self.year = 0
        self.population = data['initial_population']
        self.resources = data['initial_resources']
        self.ship_capacity = data ['ship_capacity']
        self.initial_birth_rate = data['birth_rate']
        self.initial_death_rate = data['death_rate']
        self.initial_resource_gen_rate = data['resource_gen_rate']
        self.speed = data['lightspeed_fraction'] * SPEED_OF_LIGHT_KM_PER_HR
        self.total_distance = DISTANCE_TO_PROXIMA_CENTAURI
        self.total_years = (self.total_distance / (self.speed * 24 * 365))  # Travel years
        self.distance_covered = 0
        self.initial_health_index = data['health_index']  # Initialize health index
        self.status = "Running"
        self.diseaseOutbreakEvent = 0
        self.overCrowdingEvent = 0
        self.criticalRationingEvent = 0
        self.normalRationingEvent = 0
        self.simulation_history = []  # Add this to store simulation results

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
        if self.population > 0.8 * self.ship_capacity:
            overcrowding_penalty = (self.population / self.ship_capacity - 0.8) * 10  # Penalty based on excess
            self.health_index *= 0.90  # Reduce health index due to overcrowding
            self.resource_gen_rate *= 0.9  # Decrease resource production efficiency
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
            resource_consumption = np.random.normal(CRITICAL_CONSUMPTION_THRESHOLD, 10, size=self.population).sum()  # Reduced consumption
            resource_consumption = max(0, min(resource_consumption, self.resources))
            current_year_log.append(f"Critical resource shortage. Population reduced by prioritization ({sudden_loss} lost).")
            self.resource_gen_rate *= 0.5  # Reduced productivity
            self.criticalRationingEvent = 1
        elif resources_per_person < NORMAL_CONSUMPTION:
            # Low resources, rationing
            self.health_index *= 0.9  # Penalty for insufficient resources
            resource_consumption = np.random.normal(CRITICAL_CONSUMPTION_THRESHOLD, 10, size=self.population).sum()  # Rationed consumption
            resource_consumption = max(0, min(resource_consumption, self.resources))
            self.resource_gen_rate *= 0.9
            current_year_log.append("Rationing activated due to low resources.")
            self.normalRationingEvent = 1
        else:
            # Normal resources
            self.health_index *= 1.1  # Bonus for surplus resources
            self.resource_gen_rate *= 1  # Increase productivity
            resource_consumption = np.random.normal(NORMAL_CONSUMPTION, 10, size=self.population).sum()  # Normal consumption
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



# Flask API
ship = None

@app.route('/export-csv', methods=['GET'])
def export_csv():
    global ship
    if not ship:
        return jsonify({"error": "Simulation not initialized."}), 400
    
    if not ship.simulation_history:
        return jsonify({"error": "No simulation data available."}), 400
    
    # Generate CSV data
    csv_data = ship.get_csv()
    
    # Create the response
    response = make_response(csv_data)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=generation_ship_simulation.csv'
    
    return response

@app.route('/initialize', methods=['POST'])
def initialize():
    global ship
    data = request.json
    try:
        ship = GenerationShip(data)
        return jsonify({"message": "Simulation initialized!"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/simulate', methods=['POST'])
def simulate():
    global ship
    if not ship:
        return jsonify({"error": "Simulation not initialized."}), 400

    years = request.json.get('years', 1)
    results = []

    for _ in range(years):
        current_logs = ship.simulate_year()
        status = ship.get_status(logs=current_logs)
        results.append(status)

        if ship.status in ["Failed", "Success"]:
            break

    return jsonify(results)

@app.route('/reset', methods=['POST'])
def reset():
    global ship
    if ship:
        ship.reset()
        return jsonify({"message": "Simulation reset successfully."})
    return jsonify({"error": "No simulation to reset."}), 400

@app.route('/test', methods=['GET'])
def test_cors():
    return jsonify({"message": "CORS is working!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
