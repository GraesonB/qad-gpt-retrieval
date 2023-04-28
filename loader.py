import logging
from dotenv import load_dotenv
import os
from rich.prompt import Prompt, Confirm
from rich import print
from art import *
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from langchain.text_splitter import TokenTextSplitter
from styles import *
load_dotenv()

logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('easyocr').setLevel(logging.ERROR)

text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50)

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma"
))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_KEY"),
                model_name="text-embedding-ada-002"
            )

tprint("qadr loader")
for _ in range(3):
    print("")

while True:
    collection_name = Prompt.ask(f"Enter the name of the {blue}collection{blue_e} you would like to add to")
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_ef)
        
    except (ValueError):
        if Confirm.ask(f"{italic}{collection_name}{italic_e} {red}doesn't seem to exist{red_e}, would you like to {blue}create{blue_e} {italic}{collection_name}{italic_e}?"):
            collection = chroma_client.create_collection(name=collection_name, embedding_function=openai_ef)
        else: 
            continue

    document_dir = Prompt.ask(f"Enter the document's {blue}name and path{blue_e}")
    if not os.path.exists(document_dir):
        while True:
            document_dir = Prompt.ask(f"That file {red}doesn't exist{red_e}, please type it again")
            if not os.path.exists(document_dir):
                continue
            else:
                break

    document_name = Prompt.ask(f"Enter the {blue}document name{blue_e} (for chroma tags)")

    print("")
    print(f"Collection: {green}{collection_name}{green_e}")
    print(f"Document path: {green}{document_dir}{green_e}")
    print(f"Document name: {green}{document_name}{green_e}")
    print("")

    if Confirm.ask("Proceed?"):
        break
    else:
        continue

print("")

from helpers import *

# # Read pdf into a string
logger.info("Beginning pdf scan...")
text = get_text(document_dir)
# Chunk pdf
logger.info("Chunking text...")
chunked_text = text_splitter.split_text(text)

logger.info("Saving embeddings...")
collection.add(
    documents=chunked_text,
    metadatas=[{"paper": document_name, "chunk": i} for i in range(1, len(chunked_text)+1)],
    ids=[str(i) for i in range(1, len(chunked_text)+1)]
)
logger.info("Embeddings successfully saved, pdf can now be queried.")   

