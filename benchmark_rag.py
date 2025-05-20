import os
import pandas as pd
from typing import Dict, TypedDict, List
from langgraph.graph import Graph, StateGraph
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "rag_dataset"
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Initialize sentence transformer model
model = SentenceTransformer(EMBEDDING_MODEL)

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful assistant that generates Python scripts for the OpenROAD physical design tool. You only generate scripts. Do not include any formalities or conversational text.

**Constraints:**

* The generated Python script should use the OpenROAD Python API to perform the requested actions.
* The script must include brief comments to explain each section of the code, including the purpose of each function call 
* The script should be executable.
* Assume the user has already set up the OpenROAD environment and has access to the required libraries, LEF files, and Verilog netlists.
* Prioritize clarity and readability in the generated code.
* Use descriptive variable names.

**Output Format:**

The output should be a complete Python script enclosed within ```python and ``` tags."""

# Define state
class AgentState(TypedDict):
    prompt: str
    response: str
    retrieved_docs: List[Dict]

# Create agent
class GeminiAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-flash-preview-04-17"

    def generate_response(self, state: AgentState) -> AgentState:
        try:
            # Format retrieved documents
            context = "\n\nRelevant examples from the knowledge base:\n"
            for i, doc in enumerate(state['retrieved_docs'], 1):
                context += f"\nExample {i}:\n"
                context += f"Prompt: {doc['prompt']}\n"
                context += f"Code:\n{doc['code']}\n"
                context += "\n\n"

            # Combine system prompt, context, and user prompt
            full_prompt = f"{SYSTEM_PROMPT}\n{context}\nUser request: {state['prompt']}"
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            state['response'] = response.text
        except Exception as e:
            state['response'] = f"Error generating response: {str(e)}"
        return state

def create_workflow() -> Graph:
    agent = GeminiAgent()
    workflow = StateGraph(AgentState)
    workflow.add_node("generate", agent.generate_response)
    workflow.set_entry_point("generate")
    return workflow.compile()

def get_relevant_documents(query: str, collection_name: str = COLLECTION_NAME, top_k: int = 5):
    # Connect to Milvus
    connections.connect(
        alias="default", 
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )
    
    # Get collection
    collection = Collection(collection_name)
    collection.load()
    
    # Encode query
    query_embedding = model.encode(query)
    
    # Search
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="prompt_embedding",  # Search in prompt embeddings
        param=search_params,
        limit=top_k,
        output_fields=["code", "prompt"]  # Get both code and prompt
    )
    
    # Extract and return documents
    documents = []
    for hits in results:
        for hit in hits:
            documents.append({
                'code': hit.entity.get('code'),
                'prompt': hit.entity.get('prompt'),
                'distance': hit.distance
            })
    
    return documents

def main():
    # Load benchmark data
    print("Loading benchmark data...")
    df = pd.read_csv("data/bench_data.csv")
    total_prompts = len(df)
    print(f"Found {total_prompts} prompts to process")
    
    # Initialize workflow
    workflow = create_workflow()
    
    # Process all prompts with progress bar
    results = []
    for _, row in tqdm(df.iterrows(), total=total_prompts, desc="Processing prompts"):
        # Get relevant documents
        relevant_docs = get_relevant_documents(row['prompt'])
        
        # Run workflow with retrieved documents
        initial_state = {
            "prompt": row['prompt'],
            "response": "",
            "retrieved_docs": relevant_docs
        }
        
        result = workflow.invoke(initial_state)
        results.append({
            'prompt': result['prompt'],
            'response': result['response']
        })
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/RAG_non-thinking.csv", index=False)
    print("Results saved to results/RAG_non-thinking.csv")

if __name__ == "__main__":
    main() 