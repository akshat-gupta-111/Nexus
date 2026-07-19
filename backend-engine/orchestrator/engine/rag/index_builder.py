from dotenv import load_dotenv
import os
from llama_index.core.settings import Settings

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import NodeRelationship
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


load_dotenv()

Settings.llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    temperature=0.0,
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


with open("workflow.md", "r") as f:
    md_text = f.read()
doc = Document(text=md_text)

md_parser = MarkdownNodeParser()
parent_nodes = md_parser.get_nodes_from_documents([doc])

child_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

all_nodes = []
leaf_nodes = []

for parent in parent_nodes:
    children = child_parser.get_nodes_from_documents([Document.model_validate(parent.dict())])
    
    for child in children:
        child.relationships[NodeRelationship.PARENT] = parent.as_related_node_info()
        
    leaf_nodes.extend(children)
    
    all_nodes.append(parent)
    all_nodes.extend(children)

docstore = SimpleDocumentStore()
docstore.add_documents(all_nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)

base_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
base_index.storage_context.persist(persist_dir="./my_local_storage")