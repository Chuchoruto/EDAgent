# Agent Workflow with LangGraph

This project implements a multi-agent workflow using LangGraph for script generation and verification. The workflow consists of multiple agents working together to generate, consolidate, and verify scripts based on user prompts and RAG documents.

## Features

- Three parallel drafting agents for initial script generation
- Consolidation agent for merging and verifying drafts
- Verification agent for checking script correctness
- Corrections agent for iterative improvements
- Maximum 5 iterations for verification-correction loop
- Support for RAG document integration with Milvus
- Flexible model provider integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

3. Add your API keys to the `.env` file:
- Add your LLM provider API keys (OpenAI, Anthropic, etc.)
- Configure Milvus connection details

## Milvus Setup

1. Install Docker if you haven't already (required for Milvus)

2. Download and start Milvus standalone:
```bash
# Download the docker-compose file
wget https://github.com/milvus-io/milvus/releases/download/v2.5.10/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
sudo docker compose up -d
```

3. Set up the vector database:
```bash
python setup_milvus.py
```
This will:
- Create a new collection in Milvus
- Load your RAG data from `data/RAG_data.csv`
- Generate embeddings using all-mpnet-base-v2
- Create indexes for efficient searching

## Usage

1. Initialize the workflow:
```python
from agent_workflow import create_workflow

workflow = create_workflow()
```

2. Run the workflow with your input:
```python
initial_state = {
    "user_prompt": "Your prompt here",
    "rag_documents": [],  # Add your RAG documents here
    "draft_scripts": [],
    "consolidated_script": "",
    "verification_feedback": "",
    "iteration_count": 0,
    "is_verified": False
}

result = workflow.invoke(initial_state)
```

## Customization

- Modify the agent classes to integrate with your preferred LLM providers
- Implement RAG document fetching in the agent methods
- Adjust the verification and correction logic as needed
- Modify the maximum iteration count in the `should_continue_verification` function

## Architecture

The workflow follows this sequence:
1. User prompt triggers three drafting agents
2. Drafting agents generate scripts using RAG documents
3. Consolidation agent merges and verifies drafts
4. Verification agent checks the consolidated script
5. If verification fails, corrections agent makes improvements
6. Process repeats up to 5 times or until verification passes