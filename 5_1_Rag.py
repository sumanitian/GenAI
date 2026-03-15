from pathlib import Path
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs_quick_guide.pdf"

#Load Pdf
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

#split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = text_splitter.split_documents(docs)

#embedding model
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    )

#create vector store
# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )

# print("DOCS:", len(docs))
# print("SPLIT DOCS:", len(split_docs))

# vector_store.add_documents(documents=split_docs)

print("Ingestion Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

relevant_chunks = retriver.similarity_search(
    query="What is NodeJS?"
)

context = "\n\n".join([doc.page_content for doc in relevant_chunks])

SYSTEM_PROMPT = """
You are a helpful assistant for answering questions about NodeJS.
Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{relevant_chunks}
"""

final_system_prompt = SYSTEM_PROMPT.format(relevant_chunks=context)

print(final_system_prompt)

