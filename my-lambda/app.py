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
    embedding_response = bedrock_runtime.invoke_model(
            body=json.dumps({"inputText": query}),
            modelId="amazon.titan-embed-text-v1",
            accept="application/json",
            contentType="application/json"
        )
    query_vector = json.loads(embedding_response.get("body").read()).get("embedding")
    
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
