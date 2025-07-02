# start_clients.py
import multiprocessing
import flwr as fl
from client import get_client
from utils import get_data_loaders

def start_client(cid):
    loaders, num_classes, _ = get_data_loaders("data/laryngeal dataset")
    client = get_client(cid, loaders[cid], num_classes)
    fl.client.start_numpy_client("localhost:8080", client=client)

if __name__ == "__main__":
    processes = []
    for cid in range(3):
        p = multiprocessing.Process(target=start_client, args=(cid,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
