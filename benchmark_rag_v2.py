import os
import pandas as pd
from typing import Dict, TypedDict, List, Any
from langgraph.graph import Graph, StateGraph
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Constants
COLLECTIONS = {
    "prompt_scripts": {
        "name": "rag_prompt_scripts",
        "description": "RAG dataset with code and prompt embeddings",
        "top_k": 6
    },
    "api_docs": {
        "name": "rag_api_docs",
        "description": "RAG dataset with API documentation embeddings",
        "top_k": 40
    },
    "code_pieces": {
        "name": "rag_code_pieces",
        "description": "RAG dataset with code piece embeddings",
        "top_k": 6
    }
}
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
* You must abide by file structure shown in the **Example File Calls** below (i.e., don't assume placeholder directories)

**Example File Calls:**
libDir = Path("../Design/nangate45/lib") 
lefDir = Path("../Design/nangate45/lef") 
designDir = Path("../Design/") 
design_name = "1_synth" 
design_top_module_name = "gcd" 
verilogFile = designDir/str("1_synth.v") 
verilog_file = designDir / "1_synth.v" 
site = floorplan.findSite("FreePDK45_38x28_10R_NP_162NW_34O")  
Clock_port_name = "clk" 


**Output Format:**
The output should be a complete Python script enclosed within ```python and ``` tags."""

# Define state
class AgentState(TypedDict):
    prompt: str
    response: str
    retrieved_docs: Dict[str, List[Dict[str, Any]]]

# Create agent
class GeminiAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-flash-preview-04-17"

    def generate_response(self, state: AgentState) -> AgentState:
        try:
            # Format retrieved documents
            context = "\n\nRelevant examples from the knowledge base:\n"
            
            # Add prompt scripts examples
            if state['retrieved_docs'].get('prompt_scripts'):
                context += "\nPrompt-script Pair Examples:\n"
                for i, doc in enumerate(state['retrieved_docs']['prompt_scripts'], 1):
                    context += f"\nExample {i}:\n"
                    context += f"Prompt: {doc['prompt']}\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += "\n"
            
            # Add API docs examples
            if state['retrieved_docs'].get('api_docs'):
                context += "\nAPI Documentation Examples:\n"
                for i, doc in enumerate(state['retrieved_docs']['api_docs'], 1):
                    context += f"\nAPI {i}:\n"
                    context += f"Function: {doc['function_name']}\n"
                    context += f"Parameters: {doc['parameters']}\n"
                    context += f"Return Type: {doc['return_type']}\n"
                    context += f"Description: {doc['description']}\n"
                    context += "\n"
            
            # Add code pieces examples
            if state['retrieved_docs'].get('code_pieces'):
                context += "\nCode Piece Examples:\n"
                for i, doc in enumerate(state['retrieved_docs']['code_pieces'], 1):
                    context += f"\nCode Piece {i}:\n"
                    context += f"Description: {doc['description']}\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += "\n"

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

def get_relevant_documents(query: str, query_type: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get relevant documents from all collections based on the query.
    
    Args:
        query: The search query
        query_type: Type of query to search by (can be either "prompt" or "code")
        top_k: Number of results to return per collection
        
    Returns:
        Dictionary containing results from each collection type
    """
    # Connect to Milvus
    connections.connect(
        alias="default", 
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )
    
    query_embedding = model.encode(query)
    
    # Search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    results = {}
    
    # Search prompt-scripts collection
    prompt_collection = Collection(COLLECTIONS["prompt_scripts"]["name"])
    prompt_collection.load()
    prompt_results = prompt_collection.search(
        data=[query_embedding],
        anns_field= "prompt_embedding" if query_type == "prompt" else "code_embedding",
        param=search_params,
        limit=COLLECTIONS["prompt_scripts"]["top_k"],
        output_fields=["code", "prompt"]
    )
    results["prompt_scripts"] = [
        {
            "code": hit.entity.get("code"),
            "prompt": hit.entity.get("prompt"),
            "distance": hit.distance
        }
        for hits in prompt_results
        for hit in hits
    ]
    
    # Search api-docs collection
    api_collection = Collection(COLLECTIONS["api_docs"]["name"])
    api_collection.load()
    api_results = api_collection.search(
        data=[query_embedding],
        anns_field= "description_embedding" if query_type == "prompt" else "function_name_embedding",
        param=search_params,
        limit=COLLECTIONS["api_docs"]["top_k"],
        output_fields=["function_name", "parameters", "return_type", "description"]
    )
    results["api_docs"] = [
        {
            "function_name": hit.entity.get("function_name"),
            "parameters": hit.entity.get("parameters"),
            "return_type": hit.entity.get("return_type"),
            "description": hit.entity.get("description"),
            "distance": hit.distance
        }
        for hits in api_results
        for hit in hits
    ]
    
    # Search code-pieces collection
    code_collection = Collection(COLLECTIONS["code_pieces"]["name"])
    code_collection.load()
    code_results = code_collection.search(
        data=[query_embedding],
        anns_field= "description_embedding" if query_type == "prompt" else "code_embedding",
        param=search_params,
        limit=COLLECTIONS["code_pieces"]["top_k"],
        output_fields=["code", "description"]
    )
    results["code_pieces"] = [
        {
            "code": hit.entity.get("code"),
            "description": hit.entity.get("description"),
            "distance": hit.distance
        }
        for hits in code_results
        for hit in hits
    ]
    
    return results


def main():
    # Load benchmark data
    print("Loading benchmark data...")
    #df = pd.read_csv("data/bench_data.csv")
    df = pd.read_csv("data/new_RAG_data/bench_data_v2.csv", nrows=2)
    total_prompts = len(df)
    print(f"Found {total_prompts} prompts to process")
    
    # Initialize workflow
    workflow = create_workflow()
    
    # Process all prompts with progress bar
    results = []
    for _, row in tqdm(df.iterrows(), total=total_prompts, desc="Processing prompts"):
        # Get relevant documents using prompt-based search
        relevant_docs = get_relevant_documents(row['Prompt'], query_type="prompt")
        
        # Run workflow with retrieved documents
        initial_state = {
            "prompt": row['Prompt'],
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
    results_df.to_csv("reproduced-results/RAG_Only.csv", index=False)
    print("Results saved to reproduced-results/RAG_Only.csv")

if __name__ == "__main__":
    main() 