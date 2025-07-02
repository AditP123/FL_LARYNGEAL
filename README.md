


## Directory Structure

Expected layout after extracting the dataset:

```
laryngeal_fl/
├── data/
│   └── laryngeal dataset/
│       ├── FOLD 1/
│       │   ├── Hbv/
│       │   ├── He/
│       │   ├── IPCL/
│       │   └── Le/
│       ├── FOLD 2/
│       │   └── same class folders...
│       └── FOLD 3/
│           └── same class folders...
├── client.py
├── server.py
├── model.py
├── utils.py
├── dashboard.py
├── start_clients.py
├── requirements.txt
└── metrics.csv (generated automatically)
```

Each `FOLD` acts as one FL client, and each subfolder (e.g., `Hbv`, `He`, etc.) contains PNG images for a specific class.

---

## Setup Instructions

1. **Install dependencies**

   Create a virtual environment (optional but recommended), then install:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit dashboard (in a separate terminal)**

   ```bash
   streamlit run dashboard.py
   ```

   This will show live accuracy/loss from the training.

3. **Start the FL server**

   ```bash
   python server.py
   ```

4. **Start the FL clients**

   This will simulate 3 clients in parallel (one per FOLD):

   ```bash
   python start_clients.py
   ```

---

## Optional: Bootstrapping Dashboard (Before Training)

To test the Streamlit dashboard before any actual training, you can create a `metrics.csv` file with some dummy rows:

```csv
round,client_id,accuracy,loss
1,0,0.62,0.95
1,1,0.58,1.02
1,2,0.65,0.90
```

Once real training starts, new rows will be appended to this file.

---

## Clean Up

To reset and start fresh, just delete `metrics.csv`:

```bash
rm metrics.csv
```
## Dataset Source

- Kaggle: [Laryngeal Dataset](https://www.kaggle.com/datasets/mahdiehhajian/laryngeal-dataset)
