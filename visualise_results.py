import matplotlib.pyplot as plt
import numpy as np

# --- Our Final Results ---
centralized_accuracy = 96.97
federated_accuracy = 90.15

# Data for the FL accuracy per round (from your log)
fl_rounds_history = {
    'accuracy': [70.83, 84.09, 83.71, 89.39, 87.12, 91.28, 90.53, 89.77, 92.42, 90.15]
}
rounds = np.arange(1, 11)
fl_accuracy_per_round = [acc for acc in fl_rounds_history['accuracy']]

# --- Plot 1: Final Accuracy Comparison (Bar Chart) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(8, 6))

methods = ['Centralized Learning', 'Federated Learning']
accuracies = [centralized_accuracy, federated_accuracy]
colors = ['#007ACC', '#FF4500']

bars = ax1.bar(methods, accuracies, color=colors)
ax1.set_ylabel('Final Validation Accuracy (%)')
ax1.set_title('Centralized vs. Federated Learning Performance')
ax1.set_ylim(0, 110) # Set y-axis limit to go up to 110 for better visualization

# Add accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('centralized_vs_federated.png') # Save the plot as an image
# plt.show() # Uncomment to display the plot directly


# --- Plot 2: Federated Learning Progression (Line Chart) ---
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(rounds, fl_accuracy_per_round, marker='o', linestyle='-', color='#FF4500', label='Federated Model Accuracy')
ax2.set_xlabel('Federated Learning Round')
ax2.set_ylabel('Aggregated Validation Accuracy (%)')
ax2.set_title('Federated Learning Model Performance Over Rounds')
ax2.set_xticks(rounds)
ax2.set_ylim(65, 100)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('federated_learning_progression.png') # Save the plot as an image
plt.show() # Display both plots