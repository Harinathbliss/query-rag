import json
from qdrant_client import QdrantClient
import boto3

client = QdrantClient(
    host="52.90.213.197",
    port=6333
)

collection_name = "pdf_knowledge_base"
bedrock_runtime = boto3.client(service_name="bedrock-runtime")


def lambda_handler(event,context):
    print("Lambda Invoked")
    request_body = json.loads(event.get('body')) or {}
    user_id = request_body.get('userId')
    query = request_body.get('query')
    body = json.dumps({
        "texts": [query],
        "input_type": "search_query", # సెర్చ్ కోసం కాబట్టి 'search_query'
        "truncate": "NONE"
    })
    

    embedding_response = bedrock_runtime.invoke_model(
            body=body,
            modelId="cohere.embed-english-v3",
            accept="application/json",
            contentType="application/json"
        )
    response_json = json.loads(embedding_response.get("body").read())
    query_vector = response_json.get("embeddings")[0]
    
    print("Query Vector",query_vector)

    results = client.search(
        query_vector=query_vector,
        collection_name=collection_name,
        limit=5,
        
    )

    print("User Id",user_id)
    print("Query",query)

    print("results",results)

    
    return {
    "statusCode": 200,
    "body": json.dumps(results)
    }
