import json
import boto3
from botocore.config import Config
from groq import Groq
from qdrant_client import QdrantClient
import redis

# ----------------------------
# Qdrant setup
# ----------------------------
client = QdrantClient(
    host="172.31.34.235",
    port=6333,
    check_compatibility=False
)

collection_name = "my_pdf_collection"

# ----------------------------
# Bedrock (ONLY for embeddings)
# ----------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    config=Config(retries={"max_attempts": 3})
)

# ----------------------------
# Groq client (LLM)
# ----------------------------
groq_client = Groq(
    api_key="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"
)

# ----------------------------
# Lambda handler
# ----------------------------
def lambda_handler(event, context):
    print("Lambda Invoked")

    request_body = json.loads(event.get("body") or "{}")

    redis_client = redis.Redis(
        host="tell-me-cache-sw3e3a.serverless.use1.cache.amazonaws.com",
        port=6379,
        decode_responses=True
    )
    print(f"Redis Connected {redis_client}")
    session_id = request_body.get("sessionId")
    user_id = request_body.get("userId")
    query = request_body.get("query")
    redis_key = f"chat:session:{session_id}:user{user_id}"
    #redis_client.xadd(redis_key,{"role":"User","Message":query})

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps("query is required")
        }

    # ----------------------------
    # 1. EMBEDDING (IMPORTANT FIX)
    # ----------------------------
    embed_body = json.dumps({
        "texts": [query],
        "input_type": "search_query",
        "truncate": "END"
    })

    embedding_response = bedrock.invoke_model(
        modelId="cohere.embed-english-v3",
        body=embed_body,
        contentType="application/json",
        accept="application/json"
    )

    embedding_json = json.loads(embedding_response["body"].read())
    query_vector = embedding_json["embeddings"][0]

    print("Embedding generated")

    # ----------------------------
    # 2. QDRANT SEARCH (FIXED)
    # ----------------------------
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,   # ✅ MUST be vector
        limit=3
    ).points

    # ----------------------------
    # 3. Build context
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
    # 4. RAG prompt
    # ----------------------------
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.
If answer is not present, say "I don't know".

User Query:
{query}

Context:
{search_results}
"""

    # ----------------------------
    # 5. GROQ LLM
    # ----------------------------
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=200
    )

    answer = response.choices[0].message.content

    #redis_client.xadd(redis_key,{"role":"System","Message":answer})


    # ----------------------------
    # 6. RESPONSE
    # ----------------------------
    return {
        "statusCode": 200,
        "body": json.dumps({
            "userId": user_id,
            "query": query,
            "answer": answer
        })
    }
