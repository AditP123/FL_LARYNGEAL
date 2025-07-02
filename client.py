# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import LaryngealCNN
from utils import get_data_loaders
import flwr as fl
import csv, os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss_sum += criterion(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total

def log_metrics(round_num, client_id, acc, loss):
    file_exists = os.path.exists("metrics.csv")
    with open("metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["round", "client_id", "accuracy", "loss"])
        writer.writerow([round_num, client_id, acc, loss])

def get_client(client_id, loader, num_classes):
    model = LaryngealCNN(num_classes).to(DEVICE)

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config): return [val.cpu().numpy() for val in model.state_dict().values()]
        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(model, loader, epochs=1)
            return self.get_parameters(config), len(loader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            acc, loss = evaluate(model, loader)
            log_metrics(config["server_round"], client_id, acc, loss)
            return float(loss), len(loader.dataset), {"accuracy": float(acc)}

    return FlowerClient()
