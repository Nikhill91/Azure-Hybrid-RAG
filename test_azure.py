import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("openai_endpoint"),
)

llm = os.getenv("LLM_MODEL", "gpt-5-mini")
embedding = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

print("LLM:", llm)
print("Embedding:", embedding)
print("Endpoint:", os.getenv("openai_endpoint"))

response = client.chat.completions.create(
    model=llm,
    messages=[
        {"role": "user", "content": "Reply exactly: Azure LLM works!"}
    ],
)

print("\nLLM RESPONSE:")
print(response.choices[0].message.content)

result = client.embeddings.create(
    model=embedding,
    input="Azure RAG embedding test.",
)

print("\nEMBEDDING WORKS!")
print("Dimensions:", len(result.data[0].embedding))
