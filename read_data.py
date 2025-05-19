import pandas as pd

# Read the dataset
df = pd.read_csv('data/Flow_Dataset.csv')

# Split the dataset
bench_data = df.iloc[:30]  # First 30 entries
rag_data = df.iloc[30:]    # Remaining entries

# Save the split datasets
bench_data.to_csv('data/bench_data.csv', index=False)
rag_data.to_csv('data/RAG_data.csv', index=False)

# Display the shapes of the split datasets
print("\nDataset split complete:")
print(f"Benchmark dataset shape: {bench_data.shape}")
print(f"RAG dataset shape: {rag_data.shape}")

# Display first few rows of each dataset
print("\nFirst few rows of benchmark dataset:")
print(bench_data.head())
print("\nFirst few rows of RAG dataset:")
print(rag_data.head()) 