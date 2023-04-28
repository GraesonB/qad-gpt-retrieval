from typing import List
import easyocr
from pdf2image import convert_from_path
import numpy as np
from dotenv import load_dotenv
import openai
from logger import get_logger
import os
load_dotenv()
logger = get_logger()

openai.api_key = os.getenv("OPENAI_KEY")

logger.info('Initializing reader...')
reader = easyocr.Reader(['en']) 

def get_text(dir: str) -> str:
    if '.pdf' in dir:
        images = convert_from_path(dir)
        text = ''
        for i, image in enumerate(images):
            logger.info(f"Scanning image {i+1}/{len(images)}")
            image_np = np.asarray(image)
            result = reader.readtext(image_np, detail=0, paragraph=True, rotation_info=[90, 180 ,270])
            for paragraph in result:
                paragraph = paragraph + "\n"
                text += paragraph
                
        return text

    else:
        result = reader.readtext(dir, detail=0, paragraph=True, rotation_info=[90, 180 ,270])

        text = ''
        for paragraph in result:
            paragraph = paragraph + "\n"
            text += paragraph
        return text

def overlapping_chars(str1: str, str2: str) -> str:
    overlap = ""
    for i in range(1, min(len(str1), len(str2)) + 1):
        if str1[-i:] == str2[:i]:
            overlap = str1[-i:]
    return overlap

def remove_overlap_and_combine(str1: str, str2: str, overlap: str) -> str:
    removed = str2.replace(overlap, "")
    return str1 + " " + removed

def combine_neighbors(str1: str, str2: str) -> List[str]:
    overlap = overlapping_chars(str1, str2)
    if overlap == "" or overlap == " ":
        overlap = overlapping_chars(str2, str1)
        if overlap == "" or overlap == " ":
          print(False)
          return False
        return remove_overlap_and_combine(str2, str1, overlap)
    else:
        return remove_overlap_and_combine(str1, str2, overlap)
    
def group_consecutive_chunks(dicts_list):
    indexed_list = [(i, d) for i, d in enumerate(dicts_list)]
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1]['chunk'])
    
    result = []
    current_group = []

    for i, (index, d) in enumerate(sorted_indexed_list):
        if not current_group or d['chunk'] == sorted_indexed_list[i - 1][1]['chunk'] + 1:
            current_group.append(index)
        else:
            result.append(current_group)
            current_group = [index]

        if i == len(sorted_indexed_list) - 1:
            result.append(current_group)
    return result

def combine_chunks(docs, grouped_index_list):
    combined_texts = []
    for group in grouped_index_list:
        if len(group) > 1:
            num_combos = len(group) -1
            latest_combo = ""
            for i in range(num_combos):
                if i == 0:
                    latest_combo = combine_neighbors(docs[group[i]], docs[group[i+1]])
                    if latest_combo == False:
                        logger.warning("Combination failed, appending neighbors instead...")
                        latest_combo = docs[group[i]] + docs[group[i+1]]
                else:
                    latest_combo = combine_neighbors(latest_combo, docs[group[i+1]])
                    if latest_combo == False:
                        logger.warning("Combination failed, appending neighbors instead...")
                        latest_combo = latest_combo + docs[group[i+1]]
            combined_texts.append(latest_combo)
        else:
            combined_texts.append(docs[group[0]])
    return combined_texts 
    
    
def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        input="Your text string goes here",
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


def craft_question_prompt(question: str, document_name: str, document_data: str,):
    return [
        {
            "role": "system",
            "content": f"The following contains information about a document called {document_name}. Please answer the user's question using the following information: \n \n {document_data}"
        },
        {
            "role": "user",
            "content": question
        }
    ]