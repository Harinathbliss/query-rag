import json
from qdrant_client import QdrantClient
import boto3

client = QdrantClient(
    host="18.232.130.187",
    port=6333
)

collection_name = "my_pdf_collection"
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

    results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2
    ).points

    res_collections = []
    for k in results:
        res_collections.append({
            "text":k.payload['text']
        })
    
    print("Res Collections",res_collections)

    search_results = ",".join([j['text'][:1000] for j in res_collections])
    
    print("Search Results", search_results)

    system_prompt = f"""You are an Agent Analyze the user query {query} based on the search results {search_results}"""

    print("User Id",user_id)
    print("Query",query)
    print("results",results)
    
    user_prompt = json.dumps({
    "prompt": system_prompt,
    "max_gen_len": 512,
    "temperature": 0,
    "top_p": 0.9
    })
    
    response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
    body=user_prompt
    )

    response_body = json.loads(response.get('body').read())
    answer = response_body.get('generation')

    
    return {
    "statusCode": 200,
    "body": json.dumps(answer)
    }
