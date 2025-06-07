import os
from typing import List, Dict, Any
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Constants
COLLECTIONS = {
    "prompt_scripts": {
        "name": "rag_prompt_scripts",
        "description": "RAG dataset with code and prompt embeddings",
        "top_k": 1
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
model = SentenceTransformer(EMBEDDING_MODEL)



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




if __name__ == "__main__":
    #test_query = "example prompt"
    test_query = "Given a placed design with a clk port called 'clk_i', read the technology file and cell libraries, and set the clock period as 20 ns. Perform floorplan and set the bottom-left location of the bounding box of the die as 0,0 and the top-right corner as 70,70. Set the bottom-left corner of the core's bounding box as 6,6 and the top-right corner as 64,64. After floorplanning, place the macros and the standard cells. Place the macros with a bounding box with the bottom-left corner as 32 um,32 um, and the top-right corner as 55 um,60 um. And place each macro at least 5 um to each other, and set a halo region around each macro as 5 um. Set the iteration of the global router as 10 times. In the detailed placement stage, set the maximum displacement at the x-axis as 0 um, and the y-axis as0. After the placement stage, dump the def file and name it 'placement.def'."
    results = get_relevant_documents(test_query, "prompt")
    
    # Print results from each collection
    for collection_type, docs in results.items():
        print(f"\nResults from {collection_type}:")
        for doc in docs:
            print(f"Distance: {doc['distance']}")
            for key, value in doc.items():
                if key != 'distance' and key != 'code':
                    print(f"{key}: {value}")
            print("--------------------------------")