from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "insurance-docs-index"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete if exists
if PINECONE_INDEX_NAME in [i["name"] for i in pc.list_indexes()]:
    pc.delete_index(PINECONE_INDEX_NAME)
    print(f"Deleted old index: {PINECONE_INDEX_NAME}")

# Create dense vector index
pc.create_index(
    name=PINECONE_INDEX_NAME,
    dimension=768,  # for dense embeddings
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
print(f"✅ Created dense vector index: {PINECONE_INDEX_NAME}")
