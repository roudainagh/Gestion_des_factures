import os

os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


load_dotenv()


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0
)


prompt = ChatPromptTemplate.from_template("""
You are an invoice assistant.

Answer the user's question using ONLY the information provided in the context.

The context contains invoice text and invoice metadata.

You can answer questions about:
- invoice numbers
- dates
- bill-to companies
- ship-to companies or addresses
- invoice totals
- amounts
- shipping costs
- discounts
- invoice content

If the answer is not present in the context, say:
"I don't know."

Be precise and do not invent information.

Question:
{question}

Context:
{context}
""")


rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question into a concise search query for an invoice database.

Keep important information such as:
- invoice number
- date
- company name
- customer name
- amount
- total
- shipping
- discount

Do not answer the question.

Question:
{question}

Search query:
""")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }
)


def format_docs(documents):

    formatted_documents = []

    for doc in documents:

        metadata = doc.metadata

        invoice_info = f"""
Invoice code: {metadata.get("invoice_code")}
Bill to: {metadata.get("bill_to")}
Ship to: {metadata.get("ship_to")}
Date: {metadata.get("date")}
Total: {metadata.get("total")}
Shipping: {metadata.get("shipping")}
Discount: {metadata.get("discount")}
Amount: {metadata.get("amount")}

Invoice content:
{doc.page_content}
"""

        formatted_documents.append(invoice_info)

    return "\n\n--- INVOICE ---\n\n".join(formatted_documents)


rewrite_chain = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)


rag_chain = (
    {
        "context": rewrite_chain | retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


while True:

    question = input("\nAsk something (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    answer = rag_chain.invoke(question)

    print("\nAI:", answer)