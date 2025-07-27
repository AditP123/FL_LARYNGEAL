import flwr as fl
from typing import List, Tuple, Dict, Union
from flwr.common import Metrics
import numpy as np

# --- Configuration ---
NUM_ROUNDS = 10

# --- Define a function to aggregate the evaluation metrics ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Union[float, int]]:
    """An aggregation function that calculates the weighted average of all client metrics."""
    # Create a dictionary to store the aggregated metrics
    aggregated_metrics = {}
    
    # Get a list of all metric keys from the first client (e.g., 'accuracy', 'precision')
    if not metrics:
        return {}
    metric_keys = metrics[0][1].keys()

    total_examples = sum([num_examples for num_examples, _ in metrics])

    for key in metric_keys:
        # Calculate the weighted sum of the metric
        weighted_sum = sum([num_examples * m[key] for num_examples, m in metrics])
        # Calculate the weighted average
        aggregated_metrics[key] = weighted_sum / total_examples
        
    return aggregated_metrics

# Define the strategy, now including our custom metric aggregation function
strategy = fl.server.strategy.FedAvg(
    min_available_clients=3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average, # This ensures accuracy is aggregated
)

# Start the Flower server
print(f"Starting Federated Learning server for {NUM_ROUNDS} rounds...")
history = fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    # Add this line to increase the message size limit (1GB)
    grpc_max_message_length=1024 * 1024 * 1024,
)

print("Server finished.")
# The history object will now contain the aggregated metrics for each round
print("\n--- Federated Learning Final Aggregated Metrics ---")
if history.metrics_distributed:
    for metric_name, values in history.metrics_distributed.items():
        # values is a list of tuples (round, value)
        if values:
            final_value = values[-1][1]
            if metric_name == 'accuracy':
                print(f"  {metric_name.capitalize()}: {final_value * 100:.2f}%")
            else:
                # Format metric names like f1_score to F1-score
                formatted_name = metric_name.replace('_', '-').capitalize()
                print(f"  {formatted_name}: {final_value:.4f}")
else:
    print("No metrics were distributed.")