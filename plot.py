import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv("results.csv")

# Group by 'Instance'
instances = data['Instance'].unique()
plt.rcParams.update({
    'font.size': 14,         # Default text size
    'axes.titlesize': 18,    # Title size
    'axes.labelsize': 16,    # Axis label size
    'xtick.labelsize': 12,   # X-axis tick label size
    'ytick.labelsize': 12,   # Y-axis tick label size
    'legend.fontsize': 12    # Legend font size
})
# Plot each instance
plt.figure(figsize=(10, 6))
for instance in instances:
    instance_data = data[data['Instance'] == instance]
    plt.plot(
        instance_data['Temperature'], 
        instance_data['Best Solution'], 
        marker='o',  # Add points
        label=instance
    )

# Customize the plot
plt.xlabel('Temperatura')
plt.ylabel('Solução')
plt.title('Solução vs. Temperatura')
plt.legend(title='Instance', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("best_solution_plot.png", dpi=300, bbox_inches='tight')  # Save as PNG file

# Show the plot
plt.show()
