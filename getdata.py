import kagglehub

# Download latest version
path = kagglehub.dataset_download("mahdiehhajian/laryngeal-dataset")

print("Path to dataset files:", path)