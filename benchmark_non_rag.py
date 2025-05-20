import os
import pandas as pd
from typing import Dict, TypedDict
from langgraph.graph import Graph, StateGraph
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Define state
class AgentState(TypedDict):
    prompt: str
    response: str

# Create agent
class GeminiAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-flash-preview-04-17"

    def generate_response(self, state: AgentState) -> AgentState:
        try:
            print(f"\nGenerating response for prompt: {state['prompt'][:100]}...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=state['prompt']
            )
            state['response'] = response.text
            print(f"Generated response: {state['response'][:100]}...")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            state['response'] = f"Error generating response: {str(e)}"
        return state

def create_workflow() -> Graph:
    # Initialize agent
    agent = GeminiAgent()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add node
    workflow.add_node("generate", agent.generate_response)
    
    # Set entry point
    workflow.set_entry_point("generate")
    
    return workflow.compile()

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load benchmark data
    print("\nLoading benchmark data...")
    df = pd.read_csv("data/bench_data.csv")
    first_prompt = df.iloc[0]['prompt']
    print(f"First prompt loaded: {first_prompt[:100]}...")
    
    # Initialize workflow
    print("\nInitializing workflow...")
    workflow = create_workflow()
    
    # Run workflow
    initial_state = {
        "prompt": first_prompt,
        "response": ""
    }
    
    print("\nInvoking workflow...")
    result = workflow.invoke(initial_state)
    print(f"\nWorkflow result: {result}")
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame({
        'prompt': [result['prompt']],
        'response': [result['response']]
    })
    
    print(f"\nDataFrame to save:\n{results_df}")
    results_df.to_csv("results/no_RAG_non-thinking.csv", index=False)
    print("\nResults saved to results/no_RAG_non-thinking.csv")
    
    # Verify the file was created and has content
    if os.path.exists("results/no_RAG_non-thinking.csv"):
        saved_df = pd.read_csv("results/no_RAG_non-thinking.csv")
        print(f"\nVerification - Saved file contents:\n{saved_df}")
    else:
        print("\nError: File was not created!")

if __name__ == "__main__":
    main() 