# Create and activate virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# Start Milvus
Write-Host "Starting Milvus..."
Set-Location milvus
docker-compose up -d
Set-Location ..

# Run setup and data processing
Write-Host "Setting up Milvus..."
python setup_milvus_v2.py

Write-Host "Reading data..."
python read_data_v2.py

# Run benchmarks and workflow
Write-Host "Running non-RAG benchmark..."
python benchmark_non_rag.py

Write-Host "Running RAG benchmark..."
python benchmark_rag_v2.py

Write-Host "Running agent workflow..."
python agent_workflow_v2.py

Write-Host "All processes completed!" 