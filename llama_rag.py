# Import Required Packages
import chromadb
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Define File Directory
file_directory = "path/to/your/documents"  # Update to your file path

# Embedding Model
llm = Ollama(model="llama3", request_timeout=300.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

# Set Context from Target Document
documents = SimpleDirectoryReader(input_files=[file_directory]).load_data()
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("ollama")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=10)]
)

# Set General RAG Prompt Template (in Chinese)
qa_template = PromptTemplate(
    "Based on the provided context:\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Please answer the following question:\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# Create Query Engine
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

# Example Query
query = "What are the main findings of the document?"
response = query_engine.query(query)
print(response.response)
