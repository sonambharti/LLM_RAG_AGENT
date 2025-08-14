# Documentation

##  Definition
1.  **Loader** -  In this step we are gathering all the textbooks, notes, and reference materials for a subject you want an AI to become an expert in. We're simply collecting all the raw information and putting it in one place for the system to process. <br>
&nbsp;&nbsp;Loaders are crucial components that ingest and prepare data from various sources into a format that the system can understand and use for generating responses. They act as the initial step in the RAG pipeline, bridging the gap between raw data and the language model. 

2.  **Chunking** - It involves breaking down large documents into smaller, more manageable pieces called chunks. This process is crucial for efficient information retrieval and improved accuracy in RAG systems. By dividing the data, RAG can focus on retrieving only the most relevant chunks, rather than the entire document, leading to better performance and faster responses. 

3. **Embeddings** - Embeddings are numerical representations of text (words, sentences, or even entire documents) that capture semantic meaning in a high-dimensional space. These are the semantic backbone of LLMs, it allows for semantic understanding and efficient retrieval of relevant information. <br> 
&nbsp;&nbsp;They transform text, images, or other data into numerical vectors, enabling the system to measure similarity and retrieve documents that are semantically related to a user's query. This process enhances the accuracy and relevance of generated responses in RAG systems. 

4.  **Vector Databases** - Vector databases are specialized databases designed to store, index, and query high-dimensional data represented as vectors, often referred to as embeddings. <br>
&nbsp;&nbsp;They play a critical role in RAG systems by efficiently storing and retrieving the embeddings generated during the chunking and embeddings steps. This enables the system to quickly locate relevant documents and generate accurate responses to user queries.

5.  **Retrieval** - This is the step where the system uses the vector databases to retrieve the most relevant documents based on the user's query. <br>
&nbsp;&nbsp;Retrieval is a critical component of RAG systems, as it determines the quality of the generated responses. The system uses the embeddings to measure the similarity between the user's query and the documents in the database, returning the most relevant documents for the next step in the pipeline.

6.  **Generator** - This is the final step in the RAG pipeline, where the system uses the retrieved documents to generate a response to the user's query. <br>
&nbsp;&nbsp;The generator takes the retrieved documents and uses them to produce a response that is coherent, accurate, and relevant to the user's query.


## Commands to setup this project
**Install all the requirements for the project**
```
pip install -r requirements.txt
```

**Run this code**
```
python main.py
```