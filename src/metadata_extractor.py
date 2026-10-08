import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()



class InvoiceMetadata(BaseModel):
    invoice_code: str | None = Field(default=None)
    bill_to: str | None = Field(default=None)
    ship_to: str | None = Field(default=None)
    date: str | None = Field(default=None)
    total: str | None = Field(default=None)
    shipping: str | None = Field(default=None)
    discount: str | None = Field(default=None)
    amount: str | None = Field(default=None)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


structured_llm = llm.with_structured_output(InvoiceMetadata)


prompt = ChatPromptTemplate.from_template("""
You are an invoice metadata extraction system.

Extract the following information from the invoice text:

- invoice_code
- bill_to
- ship_to
- date
- total
- shipping
- discount
- amount

If a field is not present, return null.

Invoice text:

{text}
""")


def extract_metadata(text):
    messages = prompt.format_messages(text=text)

    response = structured_llm.invoke(messages)

    return response.model_dump()