from pathlib import Path
import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient

from metadata_extractor import extract_metadata


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "Gestion_des_factures"

folder_path = Path("dataset/")

# Only test 20 PDFs
MAX_FILES = 20


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL)


# Connect to existing collection
if client.collection_exists(collection_name=COLLECTION_NAME):

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

else:
    vector_store = None


# Calculate file hash
def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


# Check if file already exists
def hash_exists_in_qdrant(file_hash):

    if vector_store is None:
        return False

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "metadata.file_hash",
                    "match": {
                        "value": file_hash
                    }
                }
            ]
        },
        limit=1,
    )

    return len(points) > 0


all_documents = []

processed_files = 0


# Process PDFs
for root, _, files in os.walk(folder_path):

    print("Processing folder:", root)

    for file in files:

        if not file.lower().endswith(".pdf"):
            continue

        # Stop after 20 PDFs
        if processed_files >= MAX_FILES:
            break

        full_path = os.path.join(root, file)

        print("\nProcessing:", file)


        # Calculate hash
        file_hash = calculate_file_hash(full_path)

        print("Hash:", file_hash)


        # Check uniqueness
        if hash_exists_in_qdrant(file_hash):

            print("Skipping already indexed:", file)

            continue


        # Load PDF
        loader = PyPDFLoader(str(full_path))

        documents = loader.load()


        # Combine all pages into one text
        text = ""

        for document in documents:
            text += document.page_content + "\n"


        # Extract invoice metadata using Groq
        print("Extracting metadata...")

        invoice_metadata = extract_metadata(text)


        # Add metadata to every page
        for document in documents:

            document.metadata.update({

                "file_hash": file_hash,

                "source_file": str(full_path),

                **invoice_metadata

            })


        # Add pages to documents
        all_documents.extend(documents)

        processed_files += 1

        print("Metadata:", invoice_metadata)

        print(
            f"Processed {processed_files}/{MAX_FILES}"
        )


    if processed_files >= MAX_FILES:
        break


print(
    f"\nLoaded {len(all_documents)} PDF pages."
)


# Split documents into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)


chunks = splitter.split_documents(all_documents)


print(
    f"Created {len(chunks)} chunks."
)


# Store chunks in Qdrant

if vector_store:

    vector_store.add_documents(chunks)

else:

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )


print("\nPDFs successfully stored in Qdrant.")