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
#COLLECTION_NAME = "rag_dataset"
COLLECTIONS = {
    "prompt_scripts": {
        "name": "rag_prompt_scripts",
        "description": "RAG dataset with code and prompt embeddings"
    },
    "api_docs": {
        "name": "rag_api_docs",
        "description": "RAG dataset with API documentation embeddings"
    },
    "code_pieces": {
        "name": "rag_code_pieces",
        "description": "RAG dataset with code piece embeddings"
    }
}
DIMENSION = 768  # Dimension for all-mpnet-base-v2
EMBEDDING_MODEL = "all-mpnet-base-v2"

def connect_to_milvus():
    """Connect to Milvus server."""
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )

def create_collections():
    """Create three separate collections with the appropriate schemas."""
    for collection_info in COLLECTIONS.values():
        if utility.has_collection(collection_info["name"]):
            utility.drop_collection(collection_info["name"])
    
    collections = {}

    ####################################
    ## CREATE PROMPT SCRIPTS COLLECTION
    ####################################
    prompt_script_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="code_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="prompt_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    #schema = CollectionSchema(fields=prompt_script_fields, description=COLLECTIONS["prompt_scripts"]["description"])
    collections["prompt_scripts"] = Collection(
        name=COLLECTIONS["prompt_scripts"]["name"], 
        schema=CollectionSchema(fields=prompt_script_fields, description=COLLECTIONS["prompt_scripts"]["description"])
    )
    
    ####################################
    ## CREATE API DOCS COLLECTION
    ####################################
    api_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="function_name_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="parameters", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="return_type", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    collections["api_docs"] = Collection(
        name=COLLECTIONS["api_docs"]["name"], 
        schema=CollectionSchema(fields=api_fields, description=COLLECTIONS["api_docs"]["description"])
    )


    ####################################
    ## CREATE CODE PIECES COLLECTION
    ####################################
    code_piece_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="code_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    collections["code_pieces"] = Collection(
        name=COLLECTIONS["code_pieces"]["name"], 
        schema=CollectionSchema(fields=code_piece_fields, description=COLLECTIONS["code_pieces"]["description"])
    )

    # Create index for vector fields
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collections["prompt_scripts"].create_index(field_name="code_embedding", index_params=index_params)
    collections["prompt_scripts"].create_index(field_name="prompt_embedding", index_params=index_params)
    collections["api_docs"].create_index(field_name="description_embedding", index_params=index_params)
    collections["api_docs"].create_index(field_name="function_name_embedding", index_params=index_params)
    collections["code_pieces"].create_index(field_name="code_embedding", index_params=index_params)
    collections["code_pieces"].create_index(field_name="description_embedding", index_params=index_params)
    return collections


def load_and_process_data() -> Dict[str, pd.DataFrame]:
    """Load and process the CSV data, from all three CSV files!"""
    data = {}

    prompt_df = pd.read_csv("data/new_RAG_data/RAG_data_v2.csv")
    prompt_df.columns = ['code', 'prompt']
    # Convert all columns to string
    prompt_df = prompt_df.astype(str)
    data["prompt_scripts"] = prompt_df

    api_df = pd.read_csv("data/new_RAG_data/RAGAPIs.csv")
    api_df.columns = ['description', 'function_name', 'parameters', 'return_type']
    # Convert all columns to string
    api_df = api_df.astype(str)
    data["api_docs"] = api_df

    code_df = pd.read_csv("data/new_RAG_data/RAGCodePiece.csv")
    code_df.columns = ['description', 'code_piece']
    # Convert all columns to string
    code_df = code_df.astype(str)
    data["code_pieces"] = code_df
    return data


def generate_embeddings(texts: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True).tolist()


def insert_data(collections: Dict[str, Collection], data: Dict[str, pd.DataFrame], model: SentenceTransformer):
    """Insert data into the appropriate collections with embeddings."""

    ## Insert prompt-script data
    prompt_embeddings = generate_embeddings(data["prompt_scripts"]["prompt"].tolist(), model)
    code_embeddings = generate_embeddings(data["prompt_scripts"]["code"].tolist(), model)
    prompt_entities = [
        data["prompt_scripts"]["code"].tolist(),
        data["prompt_scripts"]["prompt"].tolist(),
        code_embeddings,
        prompt_embeddings
    ]
    collections["prompt_scripts"].insert(prompt_entities)
    collections["prompt_scripts"].flush()


    ## Insert api-docs data
    api_description_embeddings = generate_embeddings(data["api_docs"]["description"].tolist(), model)
    api_function_name_embeddings = generate_embeddings(data["api_docs"]["function_name"].tolist(), model)
    api_entities = [
        data["api_docs"]["function_name"].tolist(),
        api_function_name_embeddings,  # Changed order to match schema
        data["api_docs"]["parameters"].tolist(),
        data["api_docs"]["return_type"].tolist(),
        data["api_docs"]["description"].tolist(),
        api_description_embeddings
    ]
    collections["api_docs"].insert(api_entities)
    collections["api_docs"].flush()


    ## Insert code-pieces data
    codepiece_code_embeddings = generate_embeddings(data["code_pieces"]["code_piece"].tolist(), model)
    codepiece_description_embeddings = generate_embeddings(data["code_pieces"]["description"].tolist(), model)
    codepiece_entities = [
        data["code_pieces"]["code_piece"].tolist(),
        codepiece_code_embeddings,
        data["code_pieces"]["description"].tolist(),
        codepiece_description_embeddings
    ]
    collections["code_pieces"].insert(codepiece_entities)
    collections["code_pieces"].flush()

    # # Generate embeddings for code and prompts
    # code_embeddings = generate_embeddings(df['code'].tolist(), model)
    # prompt_embeddings = generate_embeddings(df['prompt'].tolist(), model)
    
    # # Prepare data for insertion
    # entities = [
    #     df['code'].tolist(),     # code
    #     df['prompt'].tolist(),   # prompt
    #     code_embeddings,         # code_embedding
    #     prompt_embeddings        # prompt_embedding
    # ]
    
    # # Insert data
    # collection.insert(entities)
    # collection.flush()

def setup_milvus():
    """Main function to set up Milvus database."""
    print("Connecting to Milvus...")
    connect_to_milvus()
    
    print("Creating collection...")
    collections = create_collections()
    
    print("Loading data...")
    data = load_and_process_data()                         ## TODO: change this to be "data", and it will return a dict {str: pd.DataFrame} --> DONE
    
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating and inserting embeddings...")
    insert_data(collections, data, model)                  ## TODO: change this to take in a dict of collections and a dict of dataframes --> DONE
    
    print("Creating index...")
    #collection.load()                                   ## TODO: change this to be a loop that loads each collection in "collections" dict
    for collection in collections.values():
        collection.load()

    print("Setup complete!")
    return collections


## (just a test function to verify our RAG -- actual agent will use its own search function)
## Given a code/script query --> search prompt-scripts, code-pieces, and api-docs for relevant RAG results
def test_rag_code_search(
    collections: Dict[str, Collection],
    query: str,
    top_k: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Tests RAG search using "code" as query
    Returns: dictionary of results, from each collection
    """
    # Generate embedding for query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].tolist()

    # Search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = {}

    # Search prompt-scripts (by code embedding)
    prompt_results = collections["prompt_scripts"].search(
        data = [query_embedding],
        anns_field = "code_embedding",
        param = search_params,
        limit = top_k,
        output_fields = ["code", "prompt"]
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

    # Search api-docs (by function embedding)
    api_results = collections["api_docs"].search(
        data = [query_embedding],
        anns_field = "function_name_embedding",
        param = search_params,
        limit = top_k,
        output_fields = ["function_name", "parameters", "return_type", "description"]
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

    # Search code-pieces (by code embedding)
    code_results = collections["code_pieces"].search(
        data = [query_embedding],
        anns_field = "code_embedding",
        param = search_params,
        limit = top_k,
        output_fields = ["code", "description"]
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


def test_rag_prompt_search(
    collections: Dict[str, Collection],
    query: str,
    top_k: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Tests RAG search using "prompt" as query
    Returns: dictionary of results, from each collection
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].tolist()

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = {}

    ## Search prompt-scripts (by prompt embedding)
    prompt_results = collections["prompt_scripts"].search(
        data = [query_embedding],
        anns_field = "prompt_embedding",
        param = search_params,
        limit = top_k,
        output_fields = ["code", "prompt"]
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

    ## Search api-docs (by description embedding)
    api_results = collections["api_docs"].search(
        data = [query_embedding],
        anns_field = "description_embedding",
        param = search_params,
        limit = top_k,
        output_fields = ["function_name", "parameters", "return_type", "description"]
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

    ## Search code-pieces (by description embedding)
    code_results = collections["code_pieces"].search(
        data = [query_embedding],
        anns_field = "description_embedding",
        param = search_params,
        limit = top_k,
        output_fields = ["code", "description"]
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


if __name__ == "__main__":
    collections = setup_milvus()
    
    # Testing prompt-based search
    print("Testing RAG with prompt-based search")
    prompt_query = "example prompt"
    prompt_results = test_rag_prompt_search(collections, prompt_query)
    
    for collection_type, results in prompt_results.items():
        print(f"\nResults from {collection_type}:")
        for r in results:
            print(f"Distance: {r['distance']}")
            for key, value in r.items():
                if key != 'distance':
                    print(f"{key}: {value}")
            print("--------------------------------")
    
    # Testing code-based search
    print("Testing RAG with code-based search")
    #code_query = "example code"
    code_query = "# Disconnect the nets of the instance 'input1' from the RC network # Get the design block block = design.getBlock() # Find the instance with name 'input1' inst = block.findInst('input1') # Get the pins of the instance pins = inst.getITerms() # Iterate through the pins for pin in pins: # Get the net connected to the pin net = pin.getNet() if net: # Set the RC disconnected flag for the net net.setRCDisconnected(True)"
    code_results = test_rag_code_search(collections, code_query)
    for collection_type, results in code_results.items():
        print(f"\nResults from {collection_type}:")
        for r in results:
            print(f"Distance: {r['distance']}")
            for key, value in r.items():
                if key != 'distance':
                    print(f"{key}: {value}")
            print("--------------------------------")