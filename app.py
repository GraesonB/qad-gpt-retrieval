import chromadb
from chromadb.utils import embedding_functions
import openai
import os
from dotenv import load_dotenv
from rich.prompt import Prompt, Confirm
from rich import print
from art import *
from helpers import *
from styles import *
load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')

from chromadb.config import Settings
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma"
))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_KEY"),
                model_name="text-embedding-ada-002"
            )

while True:
    collection_name = Prompt.ask(f"Enter the name of the {blue}collection{blue_e} you would like to query")
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_ef)
    except (ValueError):
         while True:
            collection_name = Prompt.ask(f"{italic}{collection_name}{italic_e} {red}doesn't seem to exist{red_e}, try again")
            try:
                collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_ef)
                break
            except (ValueError):
                continue
    document_name = Prompt.ask(f"Enter the name of the {blue}document{blue_e} you would like to search {italic}(leave blank if you want to search the whole collection){italic_e}")
    try:
        results = collection.query(
            query_texts=["What is toolformer?"],
            n_results=15,
            where={
                "paper": {
                    "$eq": document_name
                }
            }
        )
    except (chromadb.errors.NoDatapointsException):
        while True:
            document_name = Prompt.ask(f"{italic}{document_name}{italic_e} {red}doesn't seem to have any vectors associated with it{red_e}, try a different name")
            try:
                results = collection.query(
                    query_texts=["What is toolformer?"],
                    n_results=15,
                    where={
                        "paper": {
                            "$eq": document_name
                        }
                    }
                )
                break
            except (chromadb.errors.NoDatapointsException):
                continue
    print("")
    print(f"Collection: {green}{collection_name}{green_e}")
    print(f"Document name: {green}{document_name}{green_e}")
    print("")

    if Confirm.ask("Proceed?"):
        break
    else:
        continue

while True:
    print("")
    question = Prompt.ask(f"Ask a question about {blue}{document_name}{blue_e}")

    results = collection.query(
        query_texts=[question],
        n_results=13,
        where={
            "paper": {
                "$eq": document_name
            }
        }
    )
    tags = results['metadatas'][0]
    docs = results["documents"][0]
    grouped_indexes = group_consecutive_chunks(tags)
    combined_chunks = combine_chunks(docs, grouped_indexes)

    raw_combined = ""
    for text in combined_chunks:
        raw_combined += text

    prompt = craft_question_prompt(question, document_name, raw_combined)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
        )
    
    answer = response['choices'][0]['message']['content']
    token_cost = response['usage']['total_tokens']

    print("")
    print(f"{blue}User:{blue_e} {question}")
    print("")
    print(f"{blue}ChatGPT:{blue_e} {green_s}{answer}{green_s_e}")
    print(f"{magenta}token cost: {token_cost}{magenta_e}")
    print("")


