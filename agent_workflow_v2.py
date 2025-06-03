import os
import pandas as pd
from typing import Dict, TypedDict, List, Optional, Annotated, Union, Any
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from tqdm import tqdm
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import concurrent.futures
from functools import partial

# Load environment variables
load_dotenv()

# Constants
COLLECTIONS = {
    "prompt_scripts": {
        "name": "rag_prompt_scripts",
        "description": "RAG dataset with code and prompt embeddings",
        "top_k": 5
    },
    "api_docs": {
        "name": "rag_api_docs",
        "description": "RAG dataset with API documentation embeddings",
        "top_k": 20
    },
    "code_pieces": {
        "name": "rag_code_pieces",
        "description": "RAG dataset with code piece embeddings",
        "top_k": 5
    }
}
EMBEDDING_MODEL = "all-mpnet-base-v2"
MAX_VERIFICATION_ATTEMPTS = 3

# Initialize models
model = SentenceTransformer(EMBEDDING_MODEL)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# System prompts
DRAFTING_PROMPT = """You are a helpful assistant that generates Python scripts for the OpenROAD physical design tool. You only generate scripts. Do not include any formalities or conversational text.

**Constraints:**
* The generated Python script should use the OpenROAD Python API to perform the requested actions.
* The script must include brief comments to explain each section of the code, including the purpose of each function call 
* The script should be executable.
* Assume the user has already set up the OpenROAD environment and has access to the required libraries, LEF files, and Verilog netlists.
* Prioritize clarity and readability in the generated code.
* Use descriptive variable names.

**Output Format:**
The output should be a complete Python script enclosed within ```python and ``` tags."""

CONSOLIDATION_PROMPT = """You are a consolidation agent that merges and verifies Python scripts for OpenROAD. Your task is to:
1. Compare the two provided scripts
2. Identify the best parts of each script
3. Merge them into a single, correct script
4. Ensure the merged script follows all OpenROAD best practices

**Output Format:**
The output should be a complete Python script enclosed within ```python and ``` tags."""

VERIFICATION_PROMPT = """You are a verification agent that checks OpenROAD Python scripts. Your task is to:
1. Verify that all function calls are valid OpenROAD API calls
2. Ensure the script meets all requirements from the original prompt
3. Check for any potential errors or issues

**Output Format:**
If the script is valid, respond with: "VALID: [script]"
If the script needs corrections, respond with: "INVALID: [list of required corrections]"
"""

CORRECTION_PROMPT = """You are a correction agent that fixes OpenROAD Python scripts. Your task is to:
1. Review the verification feedback
2. Make the necessary corrections to the script
3. Ensure the corrected script follows all OpenROAD best practices

**Output Format:**
The output should be a complete Python script enclosed within ```python and ``` tags."""

# Define state
class AgentState(TypedDict):
    # Input fields
    original_prompt_gemini: str
    original_prompt_openai: str
    retrieved_docs: Dict[str, List[Dict[str, Any]]]    # maps {collection_name: [doc1, doc2, ...]}, where each doc is a dict with keys: {doc_fields, distance}
    verification_attempts: int
    
    # Drafting outputs
    gemini_draft: Optional[str]
    openai_draft: Optional[str]
    
    # Processing outputs
    consolidated_script: Optional[str]
    verification_result: Optional[str]
    corrections: Optional[str]
    final_script: Optional[str]

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

class GeminiDraftingAgent:
    def __init__(self):
        self.model_name = "gemini-2.5-flash-preview-04-17"

    def generate_response(self, state: AgentState) -> AgentState:
        try:
            # Format retrieved documents
            context = "\n\nRelevant examples from the knowledge base:\n"
            
            # Add prompt-scripts examples
            if state['retrieved_docs']['prompt_scripts']:
                context += "\nExample Scripts:\n"
                for i, doc in enumerate(state['retrieved_docs']['prompt_scripts'], 1):
                    context += f"\nExample {i}:\n"
                    context += f"Prompt: {doc['prompt']}\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += "\n\n"
            
            # Add API documentation
            if state['retrieved_docs']['api_docs']:
                context += "\nRelevant API Documentation:\n"
                for i, doc in enumerate(state['retrieved_docs']['api_docs'], 1):
                    context += f"\nAPI {i}:\n"
                    context += f"Function: {doc['function_name']}\n"
                    context += f"Parameters: {doc['parameters']}\n"
                    context += f"Return Type: {doc['return_type']}\n"
                    context += f"Description: {doc['description']}\n"
                    context += "\n\n"
            
            # Add code pieces
            if state['retrieved_docs']['code_pieces']:
                context += "\nRelevant Code Pieces:\n"
                for i, doc in enumerate(state['retrieved_docs']['code_pieces'], 1):
                    context += f"\nCode Piece {i}:\n"
                    context += f"Description: {doc['description']}\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += "\n\n"

            # Combine system prompt, context, and user prompt
            full_prompt = f"{DRAFTING_PROMPT}\n{context}\nUser request: {state['original_prompt_gemini']}"
            
            response = gemini_client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            state['gemini_draft'] = response.text
        except Exception as e:
            state['gemini_draft'] = f"Error generating response: {str(e)}"
        return state

class OpenAIDraftingAgent:
    def generate_response(self, state: AgentState) -> AgentState:
        try:
            # Format retrieved documents
            context = "\n\nRelevant examples from the knowledge base:\n"
            
            # Add prompt-scripts examples
            if state['retrieved_docs']['prompt_scripts']:
                context += "\nExample Scripts:\n"
                for i, doc in enumerate(state['retrieved_docs']['prompt_scripts'], 1):
                    context += f"\nExample {i}:\n"
                    context += f"Prompt: {doc['prompt']}\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += "\n\n"
            
            # Add API documentation
            if state['retrieved_docs']['api_docs']:
                context += "\nRelevant API Documentation:\n"
                for i, doc in enumerate(state['retrieved_docs']['api_docs'], 1):
                    context += f"\nAPI {i}:\n"
                    context += f"Function: {doc['function_name']}\n"
                    context += f"Parameters: {doc['parameters']}\n"
                    context += f"Return Type: {doc['return_type']}\n"
                    context += f"Description: {doc['description']}\n"
                    context += "\n\n"
            
            # Add code pieces
            if state['retrieved_docs']['code_pieces']:
                context += "\nRelevant Code Pieces:\n"
                for i, doc in enumerate(state['retrieved_docs']['code_pieces'], 1):
                    context += f"\nCode Piece {i}:\n"
                    context += f"Description: {doc['description']}\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += "\n\n"

            # Combine system prompt, context, and user prompt
            full_prompt = f"{DRAFTING_PROMPT}\n{context}\nUser request: {state['original_prompt_openai']}"
            
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {"role": "system", "content": DRAFTING_PROMPT},
                    {"role": "user", "content": f"{context}\nUser request: {state['original_prompt_openai']}"}
                ]
            )
            state['openai_draft'] = response.choices[0].message.content
        except Exception as e:
            state['openai_draft'] = f"Error generating response: {str(e)}"
        return state

class ConsolidationAgent:
    def generate_response(self, state: AgentState) -> AgentState:
        try:
            full_prompt = f"{CONSOLIDATION_PROMPT}\n\nOriginal prompt: {state['original_prompt_gemini']}\n\nGemini draft:\n{state['gemini_draft']}\n\nOpenAI draft:\n{state['openai_draft']}"
            
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=full_prompt
            )
            state['consolidated_script'] = response.text
        except Exception as e:
            state['consolidated_script'] = f"Error consolidating scripts: {str(e)}"
        return state

class VerificationAgent:
    def generate_response(self, state: AgentState) -> AgentState:
        try:
            # Get relevant documents for the consolidated script
            script_docs = get_relevant_documents(state['consolidated_script'], query_type="code")
            
            # Format retrieved documents
            context = "\n\nRelevant examples from the knowledge base:\n"
            
            # Add prompt-scripts examples
            if script_docs['prompt_scripts']:
                context += "\nExample Scripts:\n"
                for i, doc in enumerate(script_docs['prompt_scripts'], 1):
                    context += f"\nExample {i}:\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += f"Prompt: {doc['prompt']}\n"
                    context += "\n\n"
            
            # Add API documentation
            if script_docs['api_docs']:
                context += "\nRelevant API Documentation:\n"
                for i, doc in enumerate(script_docs['api_docs'], 1):
                    context += f"\nAPI {i}:\n"
                    context += f"Function: {doc['function_name']}\n"
                    context += f"Parameters: {doc['parameters']}\n"
                    context += f"Return Type: {doc['return_type']}\n"
                    context += f"Description: {doc['description']}\n"
                    context += "\n\n"
            
            # Add code pieces
            if script_docs['code_pieces']:
                context += "\nRelevant Code Pieces:\n"
                for i, doc in enumerate(script_docs['code_pieces'], 1):
                    context += f"\nCode Piece {i}:\n"
                    context += f"Description: {doc['description']}\n"
                    context += f"Code:\n{doc['code']}\n"
                    context += "\n\n"

            full_prompt = f"{VERIFICATION_PROMPT}\n\nOriginal prompt: {state['original_prompt_gemini']}\n\nScript to verify:\n{state['consolidated_script']}\n{context}"
            
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=full_prompt
            )
            state['verification_result'] = response.text
            state['retrieved_docs'] = script_docs
        except Exception as e:
            state['verification_result'] = f"Error verifying script: {str(e)}"
        return state

class CorrectionAgent:
    def generate_response(self, state: AgentState) -> AgentState:
        try:
            full_prompt = f"{CORRECTION_PROMPT}\n\nOriginal prompt: {state['original_prompt_gemini']}\n\nScript to correct:\n{state['consolidated_script']}\n\nVerification feedback:\n{state['verification_result']}"
            
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=full_prompt
            )
            state['consolidated_script'] = response.text
            state['verification_attempts'] += 1
        except Exception as e:
            state['consolidated_script'] = f"Error correcting script: {str(e)}"
        return state

def create_workflow() -> Graph:
    # Initialize agents
    gemini_drafter = GeminiDraftingAgent()
    openai_drafter = OpenAIDraftingAgent()
    consolidator = ConsolidationAgent()
    verifier = VerificationAgent()
    corrector = CorrectionAgent()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes with distinct names from state keys
    workflow.add_node("gemini_drafter", gemini_drafter.generate_response)
    workflow.add_node("openai_drafter", openai_drafter.generate_response)
    workflow.add_node("consolidator", consolidator.generate_response)
    workflow.add_node("verifier", verifier.generate_response)
    workflow.add_node("corrector", corrector.generate_response)
    
    # Add end node
    def end_workflow(state: AgentState) -> AgentState:
        # If we hit max iterations, use the last script that was verified
        if state['verification_attempts'] >= MAX_VERIFICATION_ATTEMPTS:
            state['final_script'] = state['consolidated_script']
        return state
    
    workflow.add_node("end", end_workflow)
    
    # Define edges in sequence
    workflow.add_edge("gemini_drafter", "openai_drafter")
    workflow.add_edge("openai_drafter", "consolidator")
    workflow.add_edge("consolidator", "verifier")
    
    # Add conditional edge for verification
    def should_correct(state: AgentState) -> bool:
        if state['verification_attempts'] >= MAX_VERIFICATION_ATTEMPTS:
            return False
        return "INVALID:" in state['verification_result']
    
    workflow.add_conditional_edges(
        "verifier",
        should_correct,
        {
            True: "corrector",
            False: "end"
        }
    )
    
    # Add edge from correction back to verification
    workflow.add_edge("corrector", "verifier")
    
    # Set entry point
    workflow.set_entry_point("gemini_drafter")
    
    return workflow.compile()

def process_single_prompt(prompt: str, workflow: Graph) -> Dict:
    # Get relevant documents
    relevant_docs = get_relevant_documents(prompt, query_type="prompt")
    
    # Run workflow with retrieved documents
    initial_state = {
        "original_prompt_gemini": prompt,
        "original_prompt_openai": prompt,
        "gemini_draft": None,
        "openai_draft": None,
        "consolidated_script": None,
        "verification_result": None,
        "corrections": None,
        "final_script": None,
        "retrieved_docs": relevant_docs,
        "verification_attempts": 0
    }
    
    result = workflow.invoke(initial_state)
    
    return {
        'prompt': result['original_prompt_gemini'],
        'final_script': result['final_script'] if result['final_script'] else result['consolidated_script'],
        'iterations': result['verification_attempts']
    }

def main():
    # Load benchmark data
    print("Loading benchmark data...")
    df = pd.read_csv("data/bench_data.csv")
    
    # Initialize workflow
    workflow = create_workflow()
    
    # Create a partial function with the workflow
    process_prompt = partial(process_single_prompt, workflow=workflow)
    
    # Process prompts in parallel with progress bar
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all prompts to the executor
        future_to_prompt = {executor.submit(process_prompt, row['prompt']): row['prompt'] for _, row in df.iterrows()}
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_prompt), 
                         total=len(future_to_prompt),
                         desc="Processing prompts"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error processing prompt: {e}")
    
    # Save all results
    print("\nSaving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/multi_agent_results_v2.csv", index=False)
    print("Results saved to results/multi_agent_results_v2.csv")
    
    # Print summary
    print("\nSummary:")
    print(f"Total prompts processed: {len(all_results)}")
    print(f"Average iterations per prompt: {results_df['iterations'].mean():.2f}")
    print(f"Max iterations: {results_df['iterations'].max()}")
    print(f"Min iterations: {results_df['iterations'].min()}")

if __name__ == "__main__":
    main() 