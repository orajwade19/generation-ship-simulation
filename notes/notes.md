# Output Metrics : 
- Success rate (The ship should make it to proxima centauri)
- Final Population (The ship should hold a meaningful number of people in order to succeed in settling)

# Output analysis :
This is a terminating simulation. We will analyze the final population and the success rate. A secondary goal is minimizing the ship weight, since fast travel requires massive energy transfer - so the lighter on average, the better.  

# Determining the number of runs per configuration : 
Since the simulation can vary singificantly for different configurations, we take a middle of the road configuration to first determine the average required number of runs to have enough confidence in our measurements.


1. Initial Setup and Justification
- We chose to analyze a single configuration ("steady_pressure") as our representative case

```
        "initial_population": 1000,     
        "ship_capacity": 1050,          
        "initial_resources": 100000,    
        "birth_rate": 9.4,              
        "death_rate": 3.7,               
        "resource_gen_rate": 102,        
        "lightspeed_fraction": 0.0059,
        "health_index": 100
```

- This approach is supported by Law's methodology in "Simulation Modeling and Analysis" Chapter 9, where he demonstrates statistical principles using single representative configurations



2. Preliminary Analysis (n=10)
- Sample Mean (Final Population): 
$$\bar{X}(n) = \frac{1}{n}\sum_{i=1}^{n}X_i = 7.90$$
- Sample Variance:
$$S^2(n) = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X}(n))^2 = 74.30$$
- Success Rate (proportion): 
$$\hat{p} = \frac{\text{number of successes}}{n} = 0.80$$

3. Sample Size Determination
For 95% confidence level (α = 0.10) and relative error γ = 0.05:

For Final Population:
- Required replications based on Law's formula:
$$n_r^*(\gamma) = \min\{i \geq n: \frac{t_{i-1,0.975}\sqrt{S^2(n)/i}}{|\bar{X}(n)|} \leq \gamma/(1+\gamma)\}$$
- Yielded 283 replications needed

For Success Rate:
- Required replications:
$$n = \frac{(z_{0.975})^2\hat{p}(1-\hat{p})}{h^2}$$
where h = γ × p̂
- Yielded 96 replications needed

4. Full Analysis Results (n=300)

Code snippet used
```
if __name__ == '__main__':
    # Use the steady_pressure configuration
    config = configurations["steady_pressure"]
    
    # Create parallel client for efficient execution
    client = ParallelGenerationShipClient(max_workers=10)
    
    # Run initial replications
    print("\nRunning 10 initial replications...")
    results = client.run_parallel_simulations(config, num_runs=300)
    
    # Extract metrics of interest
    metrics = [
        {
            'run_id': run.run_id,
            'final_population': run.final_population,
            'success': 1 if run.final_status == "Success" else 0
        }
        for run in results
    ]
    
    # Print individual results
    print("\nResults from each replication:")
    for m in metrics:
        print(f"Run {m['run_id']}: Population = {m['final_population']}, Success = {'Yes' if m['success'] else 'No'}")
    
    # Calculate statistics
    n = len(metrics)
    final_pops = [m['final_population'] for m in metrics]
    successes = [m['success'] for m in metrics]
    
    mean_pop = sum(final_pops) / n
    pop_variance = sum((x - mean_pop) ** 2 for x in final_pops) / (n - 1)
    pop_std = pop_variance ** 0.5
    success_rate = sum(successes) / n
    
    print("\nSummary Statistics:")
    print(f"Final Population:")
    print(f"  Mean = {mean_pop:.2f}")
    print(f"  Std Dev = {pop_std:.2f}")
    print(f"Success Rate: {success_rate * 100:.1f}%")
```
- Final Population:
  * Mean = 8.36
  * Standard Deviation = 7.42
  * Standard Error = $$\frac{s}{\sqrt{n}} = \frac{7.53}{\sqrt{1183}} = 0.43$$
- Success Rate = 84.7%

95% Confidence Intervals:
- For Final Population:
$$\bar{X}(n) \pm t_{n-1,0.975}\sqrt{\frac{S^2(n)}{n}} = 8.36 \pm 0.840$$
- For Success Rate:
$$\hat{p} \pm z_{0.975}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = 0.84 \pm 0.05 $$







-----------------------------
-----------------------------



**Summary of Results and Use of CRN**

**Why Use CRN?**  
Common random numbers (CRN) were employed to reduce noise and improve the detection of true differences between the two configurations. By providing identical random conditions for both configurations, CRN helps distinguish genuine effects of the configuration from random variability. We made sure that all random events use a different stream of random numbers, and that they remain synchronized.

**Key Results:**

- **Population:**
  - Variance Reduction with CRN: 66.5%
  - CRN (Paired t-test): t = 16.040, p = 0.0000  
  - Non-CRN (Unpaired t-test): t = 10.064, p = 0.0000  
  Both CRN and non-CRN tests show a highly significant difference in population, with CRN making this difference even clearer.

- **Success Rate:**
  - Chi-square test: χ² = 53.842, p = 0.0000  
  There is a strongly significant difference in success rates between the two configurations. This categorical outcome shows that one configuration consistently leads to more favorable results, and the extremely low p-value confirms that this is not due to chance.

- **Health Index (for Contrast):**
  - CRN (Paired t-test): t = -0.088, p = 0.9300  
  - Non-CRN (Unpaired t-test): t = 1.109, p = 0.2681  
  In contrast to population and success rate, health index shows no significant difference between configurations. This metric demonstrates that not all outcomes are affected, and that CRN clarifies where real differences do—and do not—exist.

**Overall Interpretation:**  
Using CRN substantially reduced variance in key metrics and made genuine differences easier to identify. The results show a clear and statistically significant advantage in both population and success rate for one configuration, while highlighting that no such difference exists for health index. In this way, CRN provides greater confidence that the observed differences in critical metrics are real and not simply artifacts of random variability.

Code Used : 
```
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
    def __init__(self, base_url: str = 'http://localhost:5001', max_workers: int = 10):
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
        "ship_capacity": 1050,          
        "initial_resources": 100000,    
        "birth_rate": 9.4,              
        "death_rate": 3.7,               
        "resource_gen_rate": 102,        
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    }
    
    config_b = {
        "initial_population": 1000,     
        "ship_capacity": 1050,          
        "initial_resources": 110000,    
        "birth_rate": 9.4,              
        "death_rate": 3.7,               
        "resource_gen_rate": 101.5,        
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    }
    
    # Run comparison
    comparison_result = client.run_crn_comparison(config_a, config_b, num_runs=300)
    
    # Print report
    client.print_comparison_report(comparison_result)
```


--------------------
--------------------
# Weight Analysis Findings

## Analysis Methodology
1. Calculated total operational weight as:
   - Base weight from capacity (ship_capacity * 50)
   - Plus average operational resources (Average Resources)
2. Created summary statistics grouped by configuration parameters
3. Analyzed configurations across multiple success thresholds (80%, 85%, 90%)
4. Calculated efficiency ratio (success rate / weight) to find optimal configurations

## Key Findings

1. Minimum Weight Configurations:
   - For 80-85% success rate:
     * Ship capacity: 1050
     * Initial resources: 90,000-110,000
     * Resource generation rate: 102.0
     * Weight: ~70,200 units
     * Success rate: 88-89%

   - For 90%+ success rate:
     * Ship capacity: 1125-1150
     * Initial resources: 95,000-120,000
     * Resource generation rate: 102.0
     * Weight: 75,000-77,400 units
     * Success rate: 90-91%

2. Most Efficient Configurations (Success/Weight):
   - Ship capacity: 1050 (consistently)
   - Initial resources: 95,000-115,000
   - Resource generation rate: 103.0
   - Success rate: 100%
   - Weight: ~77,800-78,000 units

3. Trade-offs:
   - Lower weight configurations (around 70,000 units) can achieve 88-89% success
   - 90%+ success requires approximately 7% more weight
   - Maximum efficiency (100% success) requires about 10% more weight than minimum viable configurations

### Visualizations
Generated visual analysis:
1. Success Rate vs Total Weight scatter plot with efficiency frontier
2. Correlation matrix showing relationships between parameters
3. Saved in 'weight_vs_success.png' and 'correlation_matrix.png'


### Code Used : 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('grid_search_results_100_runs.csv')

# Calculate total weight for each run
df['total_weight'] = df['Config_ship_capacity'] * 50 + df['Average Resources']

# Create summary DataFrame grouped by configuration
summary_weight = df.groupby(['Config_ship_capacity', 'Config_initial_resources', 'Config_resource_gen_rate']).agg({
    'total_weight': ['mean', 'std'],
    'Final Status': lambda x: (x == 'Success').mean() * 100,  # Success rate
    'Average Resources': ['mean', 'std'],
    'Final Population': ['mean', 'std']
}).round(2)

# Find configurations that achieve different success thresholds
success_thresholds = [80, 85, 90]
efficient_configs = {}

# First flatten index for easier access
summary_weight = summary_weight.reset_index()

# Rename columns for easier access
summary_weight.columns = ['ship_capacity', 'initial_resources', 'resource_gen_rate', 
                         'weight_mean', 'weight_std', 'success_rate', 
                         'avg_resources_mean', 'avg_resources_std',
                         'final_pop_mean', 'final_pop_std']

for threshold in success_thresholds:
    successful_configs = summary_weight[summary_weight['success_rate'] >= threshold]
    min_weight_config = successful_configs.nsmallest(5, 'weight_mean')
    efficient_configs[threshold] = min_weight_config

# Print results
print("\nMost Efficient Configurations by Success Rate Threshold:")
for threshold, configs in efficient_configs.items():
    print(f"\nConfigurations achieving {threshold}% success rate:")
    print("Top 5 lowest weight configurations:")
    print(configs[['ship_capacity', 'initial_resources', 'resource_gen_rate', 
                  'weight_mean', 'success_rate']])

# Calculate efficiency ratio (success rate / weight)
summary_weight['efficiency_ratio'] = summary_weight['success_rate'] / summary_weight['weight_mean']

# Find most efficient configurations overall
print("\nMost efficient configurations (success rate / weight ratio):")
top_efficient = summary_weight.nlargest(5, 'efficiency_ratio')
print(top_efficient[['ship_capacity', 'initial_resources', 'resource_gen_rate', 
                    'weight_mean', 'success_rate', 'efficiency_ratio']])

# Create scatter plot of weight vs success rate
plt.figure(figsize=(10, 6))
plt.scatter(summary_weight['weight_mean'], 
           summary_weight['success_rate'], 
           alpha=0.5)
plt.xlabel('Total Weight')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate vs Total Weight')

# Add efficiency frontier line
frontier_configs = summary_weight.sort_values('weight_mean')
plt.plot(frontier_configs['weight_mean'], 
         frontier_configs['success_rate'].rolling(window=5).max(), 
         'r-', label='Efficiency Frontier')
plt.legend()
plt.savefig('weight_vs_success.png')
plt.close()

# Create correlation matrix for key parameters
correlation_vars = ['ship_capacity', 'initial_resources', 'resource_gen_rate',
                   'weight_mean', 'success_rate']
correlation_matrix = summary_weight[correlation_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Parameter Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Save detailed results to CSV
summary_weight.to_csv('weight_analysis_results.csv', index=False)
```

## Sensitivity Analysis

#### The base configuration for sensitivity analysis is below : 

```
    base_config = {
        "initial_population": 1000,     
        "ship_capacity": 1050,          
        "initial_resources": 100000,    
        "birth_rate": 9.4,              
        "death_rate": 3.7,               
        "resource_gen_rate": 102.0,        
        "lightspeed_fraction": 0.0059,
        "health_index": 100
    }
```

#### **Summary Statistics**
- Success rates across variations range between **25%** and **100%**.
- Final populations have a mean of **9.36 ± 7.14** and are influenced by key parameters.

| **Metric**          | **Mean**     | **Standard Deviation** |
|----------------------|--------------|------------------------|
| Success Rate (%)     | 82.5         | 20.1                  |
| Final Population     | 9.36         | 7.14                  |
| Resources Mean       | 1036         | 740.5                 |

---

#### **Critical Parameters**
From the sensitivity analysis:
1. **Resource Generation Rate** has the strongest impact on:
   - Success Rate.
   - Stability of population growth.
2. **Initial Resources** also directly influence the success rate.

**Key Observations**:
- Higher **resource generation rates** consistently improve success rates to 88-100%.
- Increasing **initial resources** reduces failure rates, stabilizing outcomes.

---

#### **Visual Evidence**
The following figures were generated:
1. **Success Rate by Variation** (`success_rate_by_variation.png`)
2. **Final Population by Variation** (`final_population_by_variation.png`)

These visuals clearly illustrate the strong dependency of outcomes on **resource generation rate** and **initial resources**, validating them as the most sensitive parameters.

#### Code Used for analysis
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sensitivity analysis data file
file_path = 'sensitivity_data_20241216_182356.csv'
sensitivity_df = pd.read_csv(file_path)

# Preview the dataset
print("Dataset Preview:")
print(sensitivity_df.head())

# Step 1: Summarize Key Metrics by variation_id
summary = sensitivity_df.groupby('variation_id').agg({
    'final_population': ['mean', 'std'],
    'final_resources': ['mean', 'std'],
    'final_health': ['mean', 'std'],
    'distance_covered': ['mean', 'std'],
    'years_survived': ['mean', 'std'],
    'final_status': lambda x: (x == 'Success').mean() * 100  # Success rate as percentage
}).reset_index()

# Rename columns for easier access
summary.columns = [
    'variation_id',
    'population_mean', 'population_std',
    'resources_mean', 'resources_std',
    'health_mean', 'health_std',
    'distance_mean', 'distance_std',
    'years_mean', 'years_std',
    'success_rate'
]

print("Summary by Variation ID:")
print(summary)

# Step 2: Analyze Success Rates
success_rates = sensitivity_df.groupby('variation_id')['final_status'].apply(
    lambda x: (x == 'Success').mean() * 100
).reset_index(name='success_rate')

print("Success Rates:")
print(success_rates)

# Step 3: Visualize Sensitivity Trends
# Bar plot for success rates
plt.figure(figsize=(12, 8))
sns.barplot(data=success_rates, x='variation_id', y='success_rate', palette='viridis')
plt.xlabel('Variation ID')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Variation')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()
plt.savefig('success_rate_by_variation.png')
plt.close()

# Boxplot for final population across variations
plt.figure(figsize=(12, 8))
sns.boxplot(data=sensitivity_df, x='variation_id', y='final_population', palette='coolwarm')
plt.xlabel('Variation ID')
plt.ylabel('Final Population')
plt.title('Final Population by Variation')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout()
plt.savefig('final_population_by_variation.png')
plt.close()

# Step 4: Correlation Analysis
# Drop non-numeric columns for correlation
sensitivity_df['success_binary'] = (sensitivity_df['final_status'] == 'Success').astype(int)
numeric_df = sensitivity_df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix: Sensitivity Analysis')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Step 5: Identify Critical Parameters
# Correlations with success
critical_params = correlation_matrix['success_binary'].sort_values(ascending=False)
print("Critical Parameters (Correlation with Success):")
print(critical_params)

# Step 6: Export Results
# Save summary statistics to a CSV
summary.to_csv('sensitivity_summary.csv', index=False)
success_rates.to_csv('success_rates.csv', index=False)

print("Processed results saved as 'sensitivity_summary.csv' and 'success_rates.csv'.")

```


#### Code used for running simulations for sensitivity analysis
```
class SensitivityDataCollector(ParallelGenerationShipClient):
    def __init__(self, base_url: str = 'http://localhost:5001', max_workers: int = 10):
        super().__init__(base_url, max_workers)
        
    def collect_sensitivity_data(self, 
                               base_config: Dict, 
                               parameter_variations: Dict,
                               num_runs: int = 100,
                               crn_base_seed: int = 42) -> pd.DataFrame:
        """
        Collect data for sensitivity analysis with all metrics saved
        
        Args:
            base_config: Base configuration
            parameter_variations: Dict of parameters and test values
            num_runs: Number of runs per configuration
            crn_base_seed: Base seed for CRN
        """
        all_runs_data = []
        
        # Run base configuration
        base_results = self._run_configuration_batch(
            "base", base_config, base_config, num_runs, crn_base_seed)
        all_runs_data.extend(base_results)
        
        # Run variations
        for param, variations in parameter_variations.items():
            print(f"\nCollecting data for {param} variations...")
            
            for value in variations:
                if value == base_config[param]:
                    continue  # Skip base value, already done
                    
                # Create modified config
                test_config = base_config.copy()
                test_config[param] = value
                
                # Run batch
                variation_id = f"{param}_{value}"
                results = self._run_configuration_batch(
                    variation_id, test_config, base_config, num_runs, crn_base_seed)
                all_runs_data.extend(results)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_runs_data)
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensitivity_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")
        
        return df

    def _run_configuration_batch(self, 
                                variation_id: str,
                                config: Dict,
                                base_config: Dict,
                                num_runs: int,
                                crn_base_seed: int) -> List[Dict]:
        """Run a batch of simulations for one configuration"""
        batch_data = []
        
        for run in range(num_runs):
            current_seed = crn_base_seed + run
            
            # Initialize simulation
            response = self.session.post(
                f"{self.base_url}/initialize",
                json={"config": config, "crn_seed": current_seed}
            )
            simulation_id = response.json()['simulation_id']
            
            # Run simulation
            results = self.session.post(
                f"{self.base_url}/simulate/{simulation_id}",
                json={'years': 1000}
            ).json()
            
            # Extract comprehensive data for each run
            run_data = self._extract_run_data(
                variation_id, config, base_config, run, current_seed, results)
            batch_data.append(run_data)
            
            print(f"Completed run {run + 1}/{num_runs} for {variation_id}")
            
        return batch_data
    
    def _extract_run_data(self,
                         variation_id: str,
                         config: Dict,
                         base_config: Dict,
                         run_number: int,
                         seed: int,
                         results: List[Dict]) -> Dict:
        """Extract comprehensive data from a single run"""
        final_state = results[-1]
        
        # Calculate parameter differences from base
        param_diffs = {
            f"diff_{param}": value - base_config.get(param, 0)
            for param, value in config.items()
        }
        
        # Count critical events
        critical_events = {
            'disease_outbreaks': sum(r.get('diseaseOutbreakEvent', 0) for r in results),
            'overcrowding_events': sum(r.get('overCrowdingEvent', 0) for r in results),
            'critical_rationing': sum(r.get('criticalRationingEvent', 0) for r in results),
            'normal_rationing': sum(r.get('normalRationingEvent', 0) for r in results)
        }
        
        # Calculate resource statistics
        resource_values = [r['resources'] for r in results]
        mean_resources = np.mean(resource_values)
        resource_stats = {
            'min_resources': min(resource_values),
            'max_resources': max(resource_values),
            'mean_resources': mean_resources,
            'std_resources': np.std(resource_values),
            'total_weight': config['ship_capacity'] * 50 + mean_resources  # Base weight + average resources
        }
        
        # Build comprehensive run data
        run_data = {
            'variation_id': variation_id,
            'run_number': run_number,
            'crn_seed': seed,
            'final_status': final_state['status'],
            'years_survived': final_state['year'],
            'final_population': final_state['population'],
            'final_resources': final_state['resources'],
            'final_health': final_state['health_index'],
            'distance_covered': final_state['distance_covered'],
            **critical_events,
            **resource_stats,
            **config,
            **param_diffs
        }
        
        return run_data

```