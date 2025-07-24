import flwr as fl
from typing import List, Tuple, Dict, Union
from flwr.common import Metrics

# --- Configuration ---
NUM_ROUNDS = 10

# --- Define a function to aggregate the evaluation metrics ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Union[float, int]]:
    """An aggregation function that calculates the weighted average of client accuracies."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

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
# The history object will now contain the aggregated accuracy for each round
final_accuracy = history.metrics_distributed['accuracy'][-1][1]
print(f"\n✅ Federated Learning - Final Aggregated Accuracy: {final_accuracy * 100:.2f}%")