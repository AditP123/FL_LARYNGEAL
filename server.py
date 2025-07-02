# server.py
import flwr as fl
from utils import get_data_loaders

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5)
    )
