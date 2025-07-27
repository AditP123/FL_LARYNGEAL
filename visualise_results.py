import matplotlib.pyplot as plt
import numpy as np

# --- Final Results Data ---
# These values should be updated with the final output from your training scripts.
# Centralized metrics are from the output of `train_centralized.py`.
# Federated metrics are the final aggregated results from `server.py`.
# Client metrics would need to be captured from each client's evaluation step.
# For this example, we'll use placeholder values for clients.

results = {
    'Centralized': {'Accuracy': 97.73, 'Precision': 0.9921, 'Recall': 0.9545, 'F1-score': 0.9730, 'AUC-ROC': 0.9994},
    'Federated (Aggregated)': {'Accuracy': 90.15, 'Precision': 0.9432, 'Recall': 0.8295, 'F1-score': 0.8825, 'AUC-ROC': 0.9885},
    'Client 1 (Fold 1)': {'Accuracy': 95.08, 'Precision': 0.9653, 'Recall': 0.9470, 'F1-score': 0.9560, 'AUC-ROC': 0.9979},
    'Client 2 (Fold 2)': {'Accuracy': 95.45, 'Precision': 0.9686, 'Recall': 0.9356, 'F1-score': 0.9518, 'AUC-ROC': 0.9972},
    'Client 3 (Fold 3)': {'Accuracy': 95.08, 'Precision': 0.9728, 'Recall': 0.9470, 'F1-score': 0.9597, 'AUC-ROC': 0.9974},
}

# Data for the FL accuracy per round (from your log)
fl_rounds_history = {
    'accuracy': [70.83, 84.09, 83.71, 89.39, 87.12, 91.28, 90.53, 89.77, 92.42, 90.15]
}
rounds = np.arange(1, 11)
# Convert accuracy to percentage for the plot
fl_accuracy_per_round = fl_rounds_history['accuracy']


# --- Plot 1: Comprehensive Performance Comparison (Grouped Bar Chart) ---
plt.style.use('seaborn-v0_8-whitegrid')

metrics = list(results['Centralized'].keys())
models = list(results.keys())
n_metrics = len(metrics)
n_models = len(models)

fig, ax = plt.subplots(figsize=(18, 10))

bar_width = 0.15
index = np.arange(n_metrics)

colors = ['#007ACC', '#FF4500', '#2ca02c', '#d62728', '#9467bd']

for i, model in enumerate(models):
    model_metrics = [results[model][metric] * 100 if metric == 'Accuracy' else results[model][metric] for metric in metrics]
    # For Accuracy, we use the value directly if it's already in percent, or multiply by 100
    model_metrics_percent = []
    for j, metric_name in enumerate(metrics):
        value = results[model][metric_name]
        if metric_name == 'Accuracy':
             model_metrics_percent.append(value)
        else:
            model_metrics_percent.append(value * 100)


    bars = ax.bar(index + i * bar_width, model_metrics_percent, bar_width, label=model, color=colors[i])
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)


ax.set_ylabel('Performance (%)')
ax.set_title('Centralized vs. Federated vs. Individual Client Performance')
ax.set_xticks(index + bar_width * (n_models - 1) / 2)
ax.set_xticklabels(metrics)
ax.set_ylim(80, 102)
ax.legend(loc='best', fontsize='x-small')

plt.tight_layout()
plt.savefig('full_performance_comparison.png')


# --- Plot 2: Federated Learning Progression (Line Chart) ---
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(rounds, fl_accuracy_per_round, marker='o', linestyle='-', color='#FF4500', label='Federated Model Accuracy')
ax2.set_xlabel('Federated Learning Round')
ax2.set_ylabel('Aggregated Validation Accuracy (%)')
ax2.set_title('Federated Learning Model Performance Over Rounds')
ax2.set_xticks(rounds)
ax2.set_ylim(65, 105)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('federated_learning_progression.png')
plt.show() # Display both plots