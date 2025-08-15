# PINECONE SETUP DOCUMENT

1. Get PINECONE API KEY.
2. 1️⃣ Create Index in Pinecone Console
    - Go to Pinecone Console [https://app.pinecone.io/organizations/-OXbtZ-LcKHrTWDL06-X/projects/fa334616-c759-4ff3-9288-6f1d6662c50c/indexes].
    - Click "Create Index".
    - Fill in:
        - Index Name → insurance-docs-index
        - (must match PINECONE_INDEX_NAME in your code)
        - Dimension → 768 (HuggingFace all-mpnet-base-v2 outputs 768 dimensions)
        - Metric → cosine
        - Pod Type → p1.x1 (free tier supports 1 pod)
    - Click Create Index.