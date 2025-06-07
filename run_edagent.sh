#!/bin/bash

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Start Milvus
echo "Starting Milvus..."
cd milvus
docker-compose up -d
cd ..

# Run setup and data processing
echo "Setting up Milvus..."
python setup_milvus_v2.py

echo "Reading data..."
python read_data_v2.py

# Run benchmarks and workflow
echo "Running non-RAG benchmark..."
python benchmark_non_rag.py

echo "Running RAG benchmark..."
python benchmark_rag_v2.py

echo "Running agent workflow..."
python agent_workflow_v2.py

echo "All processes completed!" 