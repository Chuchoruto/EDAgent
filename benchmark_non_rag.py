import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Load benchmark data
    print("\nLoading benchmark data...")
    df = pd.read_csv("data/bench_data.csv")
    first_prompt = df.iloc[0]['prompt']
    print(f"First prompt loaded: {first_prompt[:100]}...")
    
    # Generate response
    print("\nGenerating response...")
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=first_prompt
    )
    
    print(f"\nGenerated response: {response.text[:100]}...")
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame({
        'prompt': [first_prompt],
        'response': [response.text]
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