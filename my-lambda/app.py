import json
from groq import Groq
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
# Groq client
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

    user_id = request_body.get("userId")
    query = request_body.get("query")

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps("query is required")
        }

    # ----------------------------
    # 1. Embedding (KEEP YOUR CURRENT METHOD OR SWAP LATER)
    # ----------------------------
    # If you still use Cohere elsewhere, plug it here.
    # For now assume you already pass vector OR replace later.

    raise_if_missing_embedding = False  # safety placeholder

    # ----------------------------
    # 2. Vector search (Qdrant)
    # ----------------------------
    results = client.query_points(
        collection_name=collection_name,
        query=query,  # IMPORTANT: replace with vector if needed
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
    # 5. GROQ LLM CALL
    # ----------------------------
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
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

    print("Answer generated")

    # ----------------------------
    # 6. Response
    # ----------------------------
    return {
        "statusCode": 200,
        "body": json.dumps({
            "userId": user_id,
            "query": query,
            "answer": answer
        })
    }