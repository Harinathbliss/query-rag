import json
import boto3
from botocore.config import Config
from qdrant_client import QdrantClient

# ----------------------------
# Qdrant setup
# ----------------------------
client = QdrantClient(
    host="54.81.245.161",
    port=6333
)

collection_name = "my_pdf_collection"

# ----------------------------
# Bedrock client
# ----------------------------
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    config=Config(
        retries={
            "max_attempts": 5,
            "mode": "adaptive"
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
    # 2. Vector search (Qdrant)
    # ----------------------------
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    ).points

    # ----------------------------
    # 3. Build context safely
    # ----------------------------
    MAX_CONTEXT_CHARS = 1200
    search_results = ""

    for r in results:
        text = r.payload.get("text", "")

        text = text[:300]

        if len(search_results) + len(text) > MAX_CONTEXT_CHARS:
            break

        search_results += text + "\n"

    print("Context built")

    # ----------------------------
    # 4. RAG Prompt
    # ----------------------------
    system_prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not in context, say "I don't know".

User Query:
{query}

Context:
{search_results}
"""

    # ----------------------------
    # 5. NOVA LITE REQUEST (CORRECT FORMAT)
    # ----------------------------
    body = {
        "inferenceConfig": {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "top_p": 0.9
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": system_prompt
                    }
                ]
            }
        ]
    }

    # ----------------------------
    # 6. Invoke Bedrock (Nova)
    # ----------------------------
    response = bedrock_runtime.invoke_model(
        modelId="us.amazon.nova-2-lite-v1:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    # ----------------------------
    # 7. Parse response (Nova format)
    # ----------------------------
    response_body = json.loads(response["body"].read())

    answer = response_body["output"]["message"]["content"][0]["text"]

    print("Answer generated")

    # ----------------------------
    # 8. Return response
    # ----------------------------
    return {
        "statusCode": 200,
        "body": json.dumps({
            "userId": user_id,
            "query": query,
            "answer": answer
        })
    }