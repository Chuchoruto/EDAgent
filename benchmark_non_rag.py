import os
import pandas as pd
from typing import Dict, TypedDict
from langgraph.graph import Graph, StateGraph
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

# Load environment variables
load_dotenv()

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

# Create agent
class GeminiAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-flash-preview-04-17"

    def generate_response(self, state: AgentState) -> AgentState:
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser request: {state['prompt']}"
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

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load benchmark data
    print("Loading benchmark data...")
    df = pd.read_csv("data/new_RAG_data/bench_data_v2.csv", nrows=2)
    total_prompts = len(df)
    print(f"Found {total_prompts} prompts to process")
    
    # Initialize workflow
    workflow = create_workflow()
    
    # Process all prompts with progress bar
    results = []
    for _, row in tqdm(df.iterrows(), total=total_prompts, desc="Processing prompts"):
        initial_state = {
            "prompt": row['Prompt'],
            "response": ""
        }
        result = workflow.invoke(initial_state)
        results.append({
            'prompt': result['prompt'],
            'response': result['response']
        })
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv("reproduced-results/Prompt-Only.csv", index=False)
    print(f"Results saved to reproduced-results/Prompt-Only.csv")

if __name__ == "__main__":
    main() 