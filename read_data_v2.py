import pandas as pd
import numpy as np

# Read the dataset
df_prompt = pd.read_excel('data/new_RAG_data/Flow-v2.xlsx', sheet_name='cross_stage_prompt')
df_code = pd.read_excel('data/new_RAG_data/Flow-v2.xlsx', sheet_name='cross_stage_code')

# This dataset has large numbers of paraphrased prompts consecutively, so we must use linear spacing to avoid bias
indices = np.linspace(0, len(df_prompt)-1, 30, dtype=int)

# Split the dataset
prompt_bench_data = df_prompt.iloc[indices] # First 30 entries as test set
prompt_rag_data = df_prompt.drop(indices)
code_bench_data = df_code.iloc[indices]  # First 30 entries as test set
code_rag_data = df_code.drop(indices)


# For milvus testing: reduce dataset size to 100
# prompt_rag_data = prompt_rag_data.iloc[:100]
# code_rag_data = code_rag_data.iloc[:100]

#print("\nBenchmark Prompts (30 evenly spaced samples):")
#print(prompt_bench_data.to_string())

# Combine code/prompt datasets into csv files
bench_data = pd.concat([code_bench_data, prompt_bench_data], axis=1)
rag_data = pd.concat([code_rag_data, prompt_rag_data], axis=1)

# Save the split datasets
bench_data.to_csv('data/new_RAG_data/bench_data_v2.csv', index=False)
rag_data.to_csv('data/new_RAG_data/RAG_data_v2.csv', index=False)

# Display the shapes of the split datasets
print("\nDataset split complete:")
print(f"Benchmark dataset shape: {bench_data.shape}")
print(f"RAG dataset shape: {rag_data.shape}")

# Display first few rows of each dataset
print("\nFirst few rows of benchmark dataset:")
print(bench_data.head())
print("\nFirst few rows of RAG dataset:")
print(rag_data.head()) 