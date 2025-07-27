# Federated Learning for Laryngeal Image Classification

This project demonstrates a Federated Learning (FL) system for classifying laryngeal images and compares its performance against a traditional centralized learning approach. The goal is to show that FL can achieve comparable accuracy while preserving data privacy. The model architecture used is EfficientNetB0 with transfer learning.

The system now includes comprehensive metrics tracking including communication overhead analysis, training time comparison, and enhanced visualization of both performance and operational efficiency metrics.

---

## Directory Structure

The project should be organized as follows. The `laryngeal_dataset` is not included in this repository and should be placed manually.

```
.
├── laryngeal_dataset/
│   ├── FOLD 1/
│   │   ├── Hbv/
│   │   ├── He/
│   │   ├── IPCL/
│   │   └── Le/
│   ├── FOLD 2/
│   │   └── ...
│   └── FOLD 3/
│       └── ...
├── client.py
├── prepare_data.py
├── server.py
├── train_centralized.py
├── visualise_results.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/pushpavalliPI/FL_LARYNGEAL
    cd FL_LARYNGEAL
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add the Dataset**
    - Download and place the `laryngeal_dataset` directory into the root of the project folder as shown in the directory structure above.

---

## How to Run the Experiments

Follow these steps in order to replicate the results.

### Step 1: Run the Centralized Benchmark

This script trains the model on the entire dataset at once to establish the best-case performance benchmark.

```bash
python train_centralized.py
```
The script will print the final validation accuracy, model size, and training time metrics, which serve as our benchmark.

### Step 2: Run the Federated Learning Simulation

This requires **four separate terminal windows** (with the virtual environment activated in each).

1.  **In Terminal 1, start the server:**
    ```bash
    python server.py
    ```
    The server will start and wait for 3 clients to connect. It will track and aggregate communication overhead and training time metrics across all rounds.

2.  **In Terminals 2, 3, and 4, start one client each:**
    ```bash
    # In Terminal 2
    python client.py --fold 1

    # In Terminal 3
    python client.py --fold 2

    # In Terminal 4
    python client.py --fold 3
    ```
    Once all clients connect, the federated training will begin. Each client will report its training time and communication overhead per round. The server will print the final federated accuracy, total training time, and total communication overhead upon completion.

### Step 3: Generate Visualizations

This script uses the results from the previous steps to generate comprehensive comparison plots. **Note:** You may need to update the hardcoded values in the results dictionary if you re-run the experiments and get different results.

```bash
python visualise_results.py
```
This will create multiple visualization files:
- `full_performance_comparison.png` - 4-panel comprehensive metrics comparison
- `full_performance_comparison_compact.png` - Single-row compact layout
- `federated_learning_progression.png` - FL accuracy progression over rounds  
- `communication_efficiency_analysis.png` - Communication overhead and training time analysis

---

## Results

The experiments yield the following key results, demonstrating the viability of the federated approach.