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
