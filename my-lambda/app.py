import json
import boto3
from qdrant_client import QdrantClient

# Qdrant setup
client = QdrantClient(
    host="3.82.4.15",
    port=6333
)

collection_name = "my_pdf_collection"

# Bedrock client
bedrock_runtime = boto3.client(service_name="bedrock-runtime")


# ----------------------------
# Lambda handler
# ----------------------------
def lambda_handler(event, context):
    print("Lambda Invoked")

    request_body = json.loads(event.get("body") or "{}")

    user_id = request_body.get("userId")
    query = request_body.get("query")

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps("Query is required")
        }

    # ----------------------------
    # 1. Get embedding
    # ----------------------------
    body = json.dumps({
        "texts": [query],
        "input_type": "search_query",
        "truncate": "END"
    })

    embedding_response = bedrock_runtime.invoke_model(
        body=body,
        modelId="cohere.embed-english-v3",
        accept="application/json",
        contentType="application/json"
    )

    response_json = json.loads(embedding_response["body"].read())
    query_vector = response_json["embeddings"][0]

    print("Query Vector generated")

    # ----------------------------
    # 2. Qdrant search (LIMITED)
    # ----------------------------
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=2   # keep small to avoid token overflow
    ).points

    # ----------------------------
    # 3. Build SAFE context (token-controlled)
    # ----------------------------
    MAX_CONTEXT_CHARS = 1200
    search_results = ""
    current_len = 0

    for r in results:
        text = r.payload.get("text", "")[:300]

        if current_len + len(text) > MAX_CONTEXT_CHARS:
            break

        search_results += text + "\n"
        current_len += len(text)

    print("Search context prepared")

    # ----------------------------
    # 4. Prompt (RAG format safe for Llama 3)
    # ----------------------------
    system_prompt = f"""
You are an AI assistant.

Answer the question using ONLY the context below.

User Query:
{query}

Context:
{search_results}
"""

    # ----------------------------
    # 5. Bedrock Llama 3 prompt format
    # ----------------------------
    user_prompt = {
        "prompt": f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{system_prompt}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        "max_gen_len": 256,
        "temperature": 0.2,
        "top_p": 0.9
    }

    # ----------------------------
    # 6. Call Llama 3
    # ----------------------------
    response = bedrock_runtime.invoke_model(
        modelId="us.meta.llama3-1-8b-instruct-v1:0",
        body=json.dumps(user_prompt),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    answer = response_body.get("generation", "")

    print("Answer generated")

    # ----------------------------
    # 7. Response
    # ----------------------------
    return {
        "statusCode": 200,
        "body": json.dumps({
            "userId": user_id,
            "query": query,
            "answer": answer
        })
    }