import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv()

Settings.llm = AzureOpenAI(
    engine="gpt-4o", model="gpt-4o", temperature=0.0,
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    deployment_name="text-embedding-3-small-sunrise",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
)

storage_context = StorageContext.from_defaults(persist_dir="./my_local_storage")

base_index = load_index_from_storage(storage_context)
assert isinstance(base_index, VectorStoreIndex), "Expected a VectorStoreIndex in storage"

base_retriever = VectorIndexRetriever(index=base_index, similarity_top_k=6)

retriever = AutoMergingRetriever(
    base_retriever, 
    storage_context,
    verbose=True
)

query_str = (
    "Our logistics core system is experiencing a data backup backlog. "
    "Retrieve only the raw logistics logs for the current cycle to check system health. "
    "Do not run any compliance validations, do not touch any financial ledgers, "
    "and do not touch the billing database. I only want the initial data gathering task node. "
    "Provide the strict JSON graph template for this single operation."
)

nodes = retriever.retrieve(query_str)

for i, node in enumerate(nodes):
    print(f"\n--- [Retrieved Node {i+1}] Score: {node.score:.4f} ---")
    print(node.text[:300] + "...") 
