import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import skewnorm

app = Flask(__name__)
CORS(app, supports_credentials=True)


# Constants
DISTANCE_TO_PROXIMA_CENTAURI = 40140000000000 # In kilometers (4.24 light-years)
SPEED_OF_LIGHT_KM_PER_HR = 1079251200
NORMAL_CONSUMPTION = 90  # Average resource consumption per person
CRITICAL_CONSUMPTION_THRESHOLD = 45  # 50% of normal consumption

# Simulation class
class GenerationShip:
    def __init__(self, data):
        self.validate_data(data)
        self.year = 0
        self.population = data['initial_population']
        self.resources = data['initial_resources']
        self.birth_rate = data['birth_rate']
        self.death_rate = data['death_rate']
        self.resource_gen_rate = data['resource_gen_rate']
        self.speed = data['lightspeed_fraction'] * SPEED_OF_LIGHT_KM_PER_HR
        self.total_distance = DISTANCE_TO_PROXIMA_CENTAURI
        self.total_years = (self.total_distance / (self.speed * SPEED_OF_LIGHT_KM_PER_HR))/(24*365)
        self.distance_covered = 0
        self.status = "Running"

    @staticmethod
    def validate_data(data):
        required_keys = ['initial_population', 'initial_resources', 'birth_rate', 'death_rate',
                         'resource_gen_rate', 'lightspeed_fraction']
        for key in required_keys:
            if key not in data or data[key] <= 0:
                raise ValueError(f"Invalid or missing value for {key}")

    def simulate_year(self):
        if self.status == "Failed" or self.status == "Success":
            return []

        # Temporary log for the current year
        current_year_log = []

        # Update population
        births = np.random.poisson(self.birth_rate * self.population / 1000)
        deaths = np.random.poisson(self.death_rate * self.population / 1000)
        self.population += births - deaths

        # Update resources
        resource_consumption = self.population * np.random.normal(NORMAL_CONSUMPTION, 10)  # Per person
        skewness = -1  # Negative for left-tailed skew
        resource_production = max(0, skewnorm.rvs(skewness, loc=self.resource_gen_rate, scale=self.resource_gen_rate * 0.1))
        self.resources += resource_production - resource_consumption

        # Check thresholds and apply conditions
        resources_per_person = self.resources / self.population if self.population > 0 else 0

        # First Threshold - Rationing
        if resources_per_person < NORMAL_CONSUMPTION:
            self.birth_rate *= 0.9  # Decrease fertility
            self.death_rate *= 1.1  # Increase death rate due to poor conditions
            current_year_log.append(f"Year {self.year}: Rationing activated due to low resources. "
                                    f"Birth rate: {self.birth_rate:.2f}, Death rate: {self.death_rate:.2f}.")

        # Second Threshold - Population Prioritization
        if resources_per_person < CRITICAL_CONSUMPTION_THRESHOLD:
            sudden_loss = int(0.1 * self.population)  # Lose 10% of the population
            self.population -= sudden_loss
            current_year_log.append(f"Year {self.year}: Critical resource shortage. Population reduced by prioritization ({sudden_loss} lost).")

        # Check fail condition
        if self.resources <= 0 or self.population <= 0:
            self.status = "Failed"
            current_year_log.append(f"Year {self.year}: Mission failed! Resources or population depleted.")
            return current_year_log

        # Update travel
        self.distance_covered += self.speed * 24 * 365  #kms travelled in 1 yr

        # Check success
        if self.distance_covered >= DISTANCE_TO_PROXIMA_CENTAURI:
            self.status = "Success";
            current_year_log.append(f"Year {self.year}: Distance covered: {DISTANCE_TO_PROXIMA_CENTAURI:.2f} km.")
            current_year_log.append(f"Year {self.year}: Successfully reached Proxima Centauri")
        else:
            current_year_log.append(f"Year {self.year}: Distance covered: {self.distance_covered:.2f} km.")

        # Increment year
        self.year += 1

        return current_year_log

    def reset(self):
        self.year = 0
        self.population = 0
        self.resources = 0
        self.distance_covered = 0
        self.status = ""

    def get_status(self, logs=None):
        return {
            "year": self.year,
            "totalYears": self.total_years,
            "population": max(0, self.population),
            "resources": max(0, self.resources),
            "distance_covered": self.distance_covered,
            "totalDistance": self.total_distance,
            "birth_rate": self.birth_rate,
            "death_rate": self.death_rate,
            "speed": self.speed,
            "resource_gen_rate": self.resource_gen_rate,
            "status": self.status,
            "log": logs if logs else [],  # Include only the logs for the current year
        }

# Flask API
ship = None

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

        if ship.status == "Failed" or ship.status == "Success":
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
