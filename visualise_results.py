import matplotlib.pyplot as plt
import numpy as np

# --- Final Results Data ---
# These values should be updated with the final output from your training scripts.
# Centralized metrics are from the output of `train_centralized.py`.
# Federated metrics are the final aggregated results from `server.py`.
# Client metrics would need to be captured from each client's evaluation step.
# UPDATE THESE VALUES WITH YOUR ACTUAL RESULTS!

results = {
    'Centralized': {
        'Accuracy': 93.56, 'Precision': 0.9457, 'Recall': 0.9242, 'F1-score': 0.9349, 'AUC-ROC': 0.9946,
        'Training Time (s)': 100.44, 'Model/Comm Overhead (MB)': 15.47  # Model size for centralized
    },
    'Federated (Aggregated)': {
        'Accuracy': 93.94, 'Precision': 0.9792, 'Recall': 0.8902, 'F1-score': 0.9325, 'AUC-ROC': 0.9943,
        'Training Time (s)': 390.78, 'Model/Comm Overhead (MB)': 928.05  # Total communication overhead
    },
    'Client 1 (Fold 1)': {
        'Accuracy': 95.45, 'Precision': 0.9881, 'Recall': 0.9432, 'F1-score': 0.9651, 'AUC-ROC': 0.9979,
        'Training Time (s)': 288.45, 'Model/Comm Overhead (MB)': 15.47  # Individual model size
    },
    'Client 2 (Fold 2)': {
        'Accuracy': 94.70, 'Precision': 0.9646, 'Recall': 0.9280, 'F1-score': 0.9459, 'AUC-ROC': 0.9965,
        'Training Time (s)': 300.16, 'Model/Comm Overhead (MB)': 15.47  # Individual model size
    },
    'Client 3 (Fold 3)': {
        'Accuracy': 96.21, 'Precision': 0.9686, 'Recall': 0.9356, 'F1-score': 0.9518, 'AUC-ROC': 0.9972,
        'Training Time (s)': 306.18, 'Model/Comm Overhead (MB)': 15.47  # Individual model size
    },
}

# Data for the FL accuracy per round (from your log)
fl_rounds_history = {
    'accuracy': [70.83, 84.09, 83.71, 89.39, 87.12, 91.28, 90.53, 89.77, 92.42, 93.94]  # Updated with final 93.94%
}
rounds = np.arange(1, 11)
# Convert accuracy to percentage for the plot
fl_accuracy_per_round = fl_rounds_history['accuracy']


# --- Plot 1: Performance Metrics Comparison (Grouped Bar Chart) ---
plt.style.use('seaborn-v0_8-whitegrid')

# Separate performance metrics from operational metrics
performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']
operational_metrics = ['Training Time (s)', 'Model/Comm Overhead (MB)']

models = list(results.keys())
n_perf_metrics = len(performance_metrics)
n_models = len(models)

# Create subplot layout
fig = plt.figure(figsize=(16, 10))  # Reduced from 20x12

# Performance metrics subplot
ax1 = plt.subplot(2, 2, (1, 2))  # Top row, spans 2 columns

bar_width = 0.14  # Slightly narrower bars
index = np.arange(n_perf_metrics)
colors = ['#007ACC', '#FF4500', '#2ca02c', '#d62728', '#9467bd']

for i, model in enumerate(models):
    model_metrics_percent = []
    for metric_name in performance_metrics:
        value = results[model][metric_name]
        if metric_name == 'Accuracy':
             model_metrics_percent.append(value)
        else:
            model_metrics_percent.append(value * 100)

    bars = ax1.bar(index + i * bar_width, model_metrics_percent, bar_width, label=model, color=colors[i])
    # Add labels on top of bars (smaller font)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.1f}', ha='center', va='bottom', fontsize=7)

ax1.set_ylabel('Performance (%)', fontsize=10)
ax1.set_title('Performance Metrics: Centralized vs. Federated vs. Individual Clients', fontsize=11)
ax1.set_xticks(index + bar_width * (n_models - 1) / 2)
ax1.set_xticklabels(performance_metrics, fontsize=9)
ax1.set_ylim(88, 102)  # Tighter y-axis range
ax1.legend(loc='best', fontsize='x-small')

# Training Time subplot
ax2 = plt.subplot(2, 2, 3)
training_times = [results[model]['Training Time (s)'] for model in models]
bars2 = ax2.bar(models, training_times, color=colors[:len(models)])
ax2.set_ylabel('Training Time (seconds)', fontsize=10)
ax2.set_title('Training Time Comparison', fontsize=11)
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.tick_params(axis='y', labelsize=9)
for i, bar in enumerate(bars2):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + max(training_times)*0.01, f'{yval:.0f}', ha='center', va='bottom', fontsize=8)

# Communication Overhead subplot
ax3 = plt.subplot(2, 2, 4)
comm_overhead = [results[model]['Model/Comm Overhead (MB)'] for model in models]
bars3 = ax3.bar(models, comm_overhead, color=colors[:len(models)])
ax3.set_ylabel('Model Size / Communication (MB)', fontsize=10)
ax3.set_title('Model Size vs Communication Overhead', fontsize=11)
ax3.tick_params(axis='x', rotation=45, labelsize=8)
ax3.tick_params(axis='y', labelsize=9)
for i, bar in enumerate(bars3):
    yval = bar.get_height()
    if yval > 0:  # Only show label if there's actual data
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + max(comm_overhead)*0.01, f'{yval:.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout(pad=2.0)  # Add more padding between subplots
plt.savefig('full_performance_comparison.png', dpi=300, bbox_inches='tight')

# --- Alternative: Compact Single-Row Layout ---
fig_compact = plt.figure(figsize=(18, 6))

# Performance metrics - wider single plot
ax_perf = plt.subplot(1, 3, (1, 2))  # Takes 2/3 of the width

for i, model in enumerate(models):
    model_metrics_percent = []
    for metric_name in performance_metrics:
        value = results[model][metric_name]
        if metric_name == 'Accuracy':
             model_metrics_percent.append(value)
        else:
            model_metrics_percent.append(value * 100)

    bars = ax_perf.bar(index + i * bar_width, model_metrics_percent, bar_width, label=model, color=colors[i])
    for bar in bars:
        yval = bar.get_height()
        ax_perf.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.1f}', ha='center', va='bottom', fontsize=8)

ax_perf.set_ylabel('Performance (%)', fontsize=11)
ax_perf.set_title('Performance Metrics Comparison', fontsize=12)
ax_perf.set_xticks(index + bar_width * (n_models - 1) / 2)
ax_perf.set_xticklabels(performance_metrics, fontsize=10)
ax_perf.set_ylim(88, 102)
ax_perf.legend(loc='best', fontsize='small')

# Combined operational metrics
ax_ops = plt.subplot(1, 3, 3)
x_pos = np.arange(len(models))
width = 0.35

# Normalize training time and communication for better comparison
norm_training = [t/max(training_times)*100 for t in training_times]
norm_comm = [c/max(comm_overhead)*100 for c in comm_overhead]

bars_time = ax_ops.bar(x_pos - width/2, norm_training, width, label='Training Time', color='#FF6B6B', alpha=0.8)
bars_comm = ax_ops.bar(x_pos + width/2, norm_comm, width, label='Model/Comm Overhead', color='#4ECDC4', alpha=0.8)

ax_ops.set_ylabel('Normalized Scale (%)', fontsize=11)
ax_ops.set_title('Operational Metrics', fontsize=12)
ax_ops.set_xticks(x_pos)
ax_ops.set_xticklabels([m.replace(' (', '\n(') for m in models], fontsize=9)
ax_ops.legend(fontsize='small')

# Add actual values as text
for i, (bar_t, bar_c) in enumerate(zip(bars_time, bars_comm)):
    ax_ops.text(bar_t.get_x() + bar_t.get_width()/2, bar_t.get_height() + 2, 
                f'{training_times[i]:.0f}s', ha='center', va='bottom', fontsize=8)
    ax_ops.text(bar_c.get_x() + bar_c.get_width()/2, bar_c.get_height() + 2, 
                f'{comm_overhead[i]:.0f}MB', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('full_performance_comparison_compact.png', dpi=300, bbox_inches='tight')


# --- Plot 2: Federated Learning Progression (Line Chart) ---
fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.plot(rounds, fl_accuracy_per_round, marker='o', linestyle='-', color='#FF4500', label='Federated Model Accuracy', linewidth=2, markersize=6)
ax2.set_xlabel('Federated Learning Round')
ax2.set_ylabel('Aggregated Validation Accuracy (%)')
ax2.set_title('Federated Learning Model Performance Over Rounds')
ax2.set_xticks(rounds)
ax2.set_ylim(65, 105)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add value labels on each point
for i, v in enumerate(fl_accuracy_per_round):
    ax2.annotate(f'{v:.1f}%', (rounds[i], v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('federated_learning_progression.png', dpi=300, bbox_inches='tight')

# --- Plot 3: New - Communication Efficiency Analysis ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 6))

# Communication overhead per model (excluding centralized)
federated_models = [model for model in models if 'Centralized' not in model]
fed_comm_overhead = [results[model]['Model/Comm Overhead (MB)'] for model in federated_models]
fed_training_time = [results[model]['Training Time (s)'] for model in federated_models]

# Plot 3a: Communication overhead (federated only)
bars3a = ax3a.bar(federated_models, fed_comm_overhead, color=['#FF4500', '#2ca02c', '#d62728', '#9467bd'][:len(federated_models)])
ax3a.set_ylabel('Communication Overhead (MB)')
ax3a.set_title('Communication Overhead - Federated Models Only')
ax3a.tick_params(axis='x', rotation=45)
for i, bar in enumerate(bars3a):
    yval = bar.get_height()
    if yval > 0:
        ax3a.text(bar.get_x() + bar.get_width()/2.0, yval + max(fed_comm_overhead)*0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

# Plot 3b: Training time
bars3b = ax3b.bar(federated_models, fed_training_time, color=['#FF4500', '#2ca02c', '#d62728', '#9467bd'][:len(federated_models)])
ax3b.set_ylabel('Training Time (seconds)')
ax3b.set_title('Training Time - Federated Models Only')
ax3b.tick_params(axis='x', rotation=45)
for i, bar in enumerate(bars3b):
    yval = bar.get_height()
    if yval > 0:
        ax3b.text(bar.get_x() + bar.get_width()/2.0, yval + max(fed_training_time)*0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('communication_efficiency_analysis.png', dpi=300, bbox_inches='tight')

plt.show()  # Display all plots

print("\n🎉 Visualization complete! Generated files:")
print("1. full_performance_comparison.png - 4-panel comprehensive view (reduced size)")
print("2. full_performance_comparison_compact.png - NEW! Single-row compact layout")
print("3. federated_learning_progression.png - FL accuracy progression over rounds")
print("4. communication_efficiency_analysis.png - Communication overhead and training time analysis")
print("\n� Use the compact version if the 4-panel layout is too large for your display!")