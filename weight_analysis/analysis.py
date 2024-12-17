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