import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "rag_dataset"
DIMENSION = 768  # Dimension for all-mpnet-base-v2
EMBEDDING_MODEL = "all-mpnet-base-v2"

def connect_to_milvus():
    """Connect to Milvus server."""
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )

def create_collection():
    """Create a new collection with the specified schema."""
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Define the collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="code_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="prompt_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields=fields, description="RAG dataset with code and prompt embeddings")
    
    # Create collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # Create index for vector fields
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="code_embedding", index_params=index_params)
    collection.create_index(field_name="prompt_embedding", index_params=index_params)
    
    return collection

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load and process the CSV data."""
    df = pd.read_csv(file_path)
    # Ensure column names are correct
    df.columns = ['code', 'prompt']
    return df

def generate_embeddings(texts: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True).tolist()

def insert_data(collection: Collection, df: pd.DataFrame, model: SentenceTransformer):
    """Insert data into the collection with embeddings."""
    # Generate embeddings for code and prompts
    code_embeddings = generate_embeddings(df['code'].tolist(), model)
    prompt_embeddings = generate_embeddings(df['prompt'].tolist(), model)
    
    # Prepare data for insertion
    entities = [
        df['code'].tolist(),     # code
        df['prompt'].tolist(),   # prompt
        code_embeddings,         # code_embedding
        prompt_embeddings        # prompt_embedding
    ]
    
    # Insert data
    collection.insert(entities)
    collection.flush()

def setup_milvus():
    """Main function to set up Milvus database."""
    print("Connecting to Milvus...")
    connect_to_milvus()
    
    print("Creating collection...")
    collection = create_collection()
    
    print("Loading data...")
    df = load_and_process_data("data/RAG_data.csv")
    
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating and inserting embeddings...")
    insert_data(collection, df, model)
    
    print("Creating index...")
    collection.load()
    
    print("Setup complete!")

def search_similar(
    collection: Collection,
    query: str,
    search_field: str = "prompt",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for similar items in the collection.
    
    Args:
        collection: Milvus collection
        query: Search query
        search_field: Field to search in ("code" or "prompt")
        top_k: Number of results to return
    
    Returns:
        List of dictionaries containing the results
    """
    # Generate embedding for the query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].tolist()
    
    # Search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    # Perform search
    results = collection.search(
        data=[query_embedding],
        anns_field=f"{search_field}_embedding",
        param=search_params,
        limit=top_k,
        output_fields=["code", "prompt"]
    )
    
    # Format results
    formatted_results = []
    for hits in results:
        for hit in hits:
            formatted_results.append({
                "code": hit.entity.get("code"),
                "prompt": hit.entity.get("prompt"),
                "distance": hit.distance
            })
    
    return formatted_results

if __name__ == "__main__":
    setup_milvus()
    
    # Example search
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # Example: Search by prompt
    print("\nSearching by prompt:")
    #prompt_query = "Given a placed design with a clk port called 'clk_i', read the technology file and cell libraries, and set the clock period as 20 ns. Perform floorplan and set the bottom-left location of the bounding box of the die as 0,0 and the top-right corner as 70,70. Set the bottom-left corner of the core's bounding box as 6,6 and the top-right corner as 64,64. After floorplanning, place the macros and the standard cells. Place the macros with a bounding box with the bottom-left corner as 32 um,32 um, and the top-right corner as 55 um,60 um. And place each macro at least 5 um to each other, and set a halo region around each macro as 5 um. Set the iteration of the global router as 10 times. In the detailed placement stage, set the maximum displacement at the x-axis as 0 um, and the y-axis as0. After the placement stage, dump the def file and name it 'placement.def'."
    prompt_query = "example prompt"
    results = search_similar(collection, prompt_query, search_field="prompt")
    print("LEN of results: ", len(results))
    for r in results:
        print(f"Code: {r['code']}")
        print(f"Prompt: {r['prompt']}")
        print(f"Distance: {r['distance']}\n")
        print("--------------------------------")
    
    # Example: Search by code
    print("\nSearching by code:")
    #code_query = "# Disconnect the nets of the instance 'input1' from the RC network # Get the design block block = design.getBlock() # Find the instance with name 'input1' inst = block.findInst('input1') # Get the pins of the instance pins = inst.getITerms() # Iterate through the pins for pin in pins: # Get the net connected to the pin net = pin.getNet() if net: # Set the RC disconnected flag for the net net.setRCDisconnected(True)"
    code_query = "example code"
    results = search_similar(collection, code_query, search_field="code")
    for r in results:
        print(f"Code: {r['code']}")
        print(f"Prompt: {r['prompt']}")
        print(f"Distance: {r['distance']}\n") 
        print("--------------------------------")