import json
import boto3
from botocore.config import Config
from qdrant_client import QdrantClient

# ----------------------------
# Qdrant setup
# ----------------------------
client = QdrantClient(
    host="3.82.4.15",
    port=6333
)

collection_name = "my_pdf_collection"

# ----------------------------
# Bedrock client (safer retries)
# ----------------------------
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    config=Config(
        retries={
            "max_attempts": 2
        }
    )
)

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
            "body": json.dumps("query is required")
        }

    # ----------------------------
    # 1. Embedding (Cohere)
    # ----------------------------
    embed_body = json.dumps({
        "texts": [query],
        "input_type": "search_query",
        "truncate": "END"
    })

    embedding_response = bedrock_runtime.invoke_model(
        modelId="cohere.embed-english-v3",
        body=embed_body,
        accept="application/json",
        contentType="application/json"
    )

    embedding_json = json.loads(embedding_response["body"].read())
    query_vector = embedding_json["embeddings"][0]

    print("Embedding generated")

    # ----------------------------
    # 2. Vector search (STRICT LIMIT)
    # ----------------------------
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1   # IMPORTANT: keep small for big models
    ).points

    # ----------------------------
    # 3. Safe context builder (token-safe)
    # ----------------------------
    MAX_CONTEXT_CHARS = 800
    search_results = ""

    for r in results:
        text = r.payload.get("text", "")

        # aggressive trim
        text = text[:250]

        if len(search_results) + len(text) > MAX_CONTEXT_CHARS:
            break

        search_results += text + "\n"

    print("Context built")

    # ----------------------------
    # 4. Prompt (clean RAG format)
    # ----------------------------
    system_prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

User Query:
{query}

Context:
{search_results}
"""

    # ----------------------------
    # 5. BIG MODEL (Claude 3 Sonnet)
    # ----------------------------
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.2,
        "top_p": 0.9,
        "messages": [
            {
                "role": "user",
                "content": system_prompt
            }
        ]
    }

    # ----------------------------
    # 6. Invoke Bedrock (BIG MODEL)
    # ----------------------------
    response = bedrock_runtime.invoke_model(
        modelId="us.anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())

    answer = response_body["content"][0]["text"]

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