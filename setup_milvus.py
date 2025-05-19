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
COLLECTION_NAME = "flow_dataset"
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
        FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="script", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="prompt_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="script_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields=fields, description="Flow dataset with embeddings")
    
    # Create collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # Create index for vector fields
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="prompt_embedding", index_params=index_params)
    collection.create_index(field_name="script_embedding", index_params=index_params)
    
    return collection

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load and process the CSV data."""
    df = pd.read_csv(file_path)
    # Ensure column names are correct
    df.columns = ['prompt', 'script']
    return df

def generate_embeddings(texts: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True).tolist()

def insert_data(collection: Collection, df: pd.DataFrame, model: SentenceTransformer):
    """Insert data into the collection with embeddings."""
    # Generate embeddings for prompts and scripts
    prompt_embeddings = generate_embeddings(df['prompt'].tolist(), model)
    script_embeddings = generate_embeddings(df['script'].tolist(), model)
    
    # Prepare data for insertion
    entities = [
        df['prompt'].tolist(),  # prompt
        df['script'].tolist(),  # script
        prompt_embeddings,      # prompt_embedding
        script_embeddings       # script_embedding
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
    df = load_and_process_data("data/Flow_Dataset.csv")
    
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
        search_field: Field to search in ("prompt" or "script")
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
        output_fields=["prompt", "script"]
    )
    
    # Format results
    formatted_results = []
    for hits in results:
        for hit in hits:
            formatted_results.append({
                "prompt": hit.entity.get("prompt"),
                "script": hit.entity.get("script"),
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
    results = search_similar(collection, "example prompt", search_field="prompt")
    for r in results:
        print(f"Prompt: {r['prompt']}")
        print(f"Script: {r['script']}")
        print(f"Distance: {r['distance']}\n")
    
    # Example: Search by script
    print("\nSearching by script:")
    results = search_similar(collection, "example script", search_field="script")
    for r in results:
        print(f"Prompt: {r['prompt']}")
        print(f"Script: {r['script']}")
        print(f"Distance: {r['distance']}\n") 