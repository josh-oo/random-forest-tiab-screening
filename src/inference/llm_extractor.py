from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-5-mini"
OPENAI_CLIENT = AsyncOpenAI()
DATA = os.environ.get("DATA")
LLM_QUESTIONS = pd.read_csv(os.path.join(DATA, "llm_questions.csv"))

class StructuredAnswer(BaseModel):
    answer: Literal["YES", "NO", "UNSURE"]

def convert_yes_no_unsure_to_int(value: str, unsure_as_1: bool) -> int:
    """Convert Yes/No/Unsure to binary: NO->0, YES->1, UNSURE->1 or 0.5"""
    if value == "NO":
        return 0
    elif value == "YES":
        return 1
    elif value == "UNSURE":
        return 1 if unsure_as_1 else 0.5
    else:
        raise ValueError(f"Unexpected value: {value}")
    
def extractions_to_binary(data):
    features = ['E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'I1', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9']
    vector = []
    for feature in features:
        vector.append(convert_yes_no_unsure_to_int(data[feature], True))
    return vector

async def call_llm(paper_title, paper_abstract, question_text):
    """
    Call OpenAI API with prompt caching.
    Structure:
    - System message (cached): Instructions
    - User message (cached): Title + Abstract
    - User message (not cached): Specific question
    """

    messages = [
        {
            "role": "system",
            "content": "You are a systematic review screening assistant. Your task is to determine if a paper meets specific inclusion or exclusion criteria based on its title and abstract. You must answer with ONLY one word: YES, NO, or UNSURE (only if you are really uncertain). Do not provide any explanation or additional text.",
        },
        {"role": "user", "content": f"Title: {paper_title}\n\nAbstract: {paper_abstract}"},
        {
            "role": "user",
            "content": f"Task: according to the paper's title and abstract, answer the following question. Only answer with YES, NO or UNSURE (just if you are really unsure). Nothing else.\n\nQuestion: {question_text}",
        },
    ]

    try:
        response = await OPENAI_CLIENT.chat.completions.parse(model=OPENAI_MODEL, messages=messages, response_format=StructuredAnswer)
        return response
    except Exception as e:
        print(f"Error processing paper {paper_title}, question {question_text}: {e}")
        return None

async def extract_features(title, abstract):
    """
    Process all questions for a single paper sequentially.
    This preserves prompt caching for the paper's title+abstract.

    Returns: (paper_id, num_processed, num_skipped)
    """

    data = {}

    async def call_for_question(question_row):
        question_id = question_row["id"]
        question_text = question_row["question"]
        response = await call_llm(title, abstract, question_text)
        return question_id, response

    tasks = [call_for_question(row) for _, row in LLM_QUESTIONS.iterrows()]
    results = await asyncio.gather(*tasks)
    total_input_tokens_used = 0
    total_output_tokens_used = 0
    for question_id, response in results:
        print(response.usage.prompt_tokens_details)
        total_input_tokens_used += response.usage.prompt_tokens
        total_output_tokens_used += response.usage.completion_tokens
        result = response.choices[0].message.parsed.answer
        data[question_id] = result

    data = extractions_to_binary(data)

    return data, total_input_tokens_used, total_output_tokens_used