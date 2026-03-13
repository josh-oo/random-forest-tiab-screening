from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import asyncio
import hashlib
import json
import os
from typing import List

from dotenv import load_dotenv

import redis.asyncio as redis

load_dotenv()

OPENAI_MODEL = os.environ.get("MODEL")
print("Model: ", OPENAI_MODEL)

DATA = os.environ.get("DATA")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CLIENT = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

LLM_QUESTIONS = pd.read_csv(os.path.join(DATA, "llm_questions.csv"))

REDIS_CACHE_PREFIX = os.environ.get("REDIS_CACHE_PREFIX", "llm-single-question")
REDIS_TTL_SECONDS = int(os.environ.get("REDIS_TTL_SECONDS", 60 * 60 * 24 * 30))

redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=int(os.environ.get("REDIS_DB", 0)),
    decode_responses=True,
)

class StructuredAnswer(BaseModel):
    evidence: str = Field(description="Supporting evidence or reasoning for the answer.")
    answer: Literal["YES", "NO", "UNSURE"] = Field(description="Final answer selection: YES, NO, or UNSURE.")

class ExclusionReason(BaseModel):
    ids: list[int] = Field(description="The ids of the corresponding reasons leading to this decision.")
    evidence: str = Field(description="Supporting evidence or reasoning for the decision.")

class StructuredMultiAnswer(BaseModel):
    i01: Literal["YES", "NO", "UNSURE"] = Field(description="Does the paper describe a randomized controlled trial (RCT), controlled clinical trial (CCT), or any trial with prospective assignment of participants?")
    i02: Literal["YES", "NO", "UNSURE"] = Field(description="Does the paper include or reference a trial registry identifier (e.g., NCT, ISRCTN, EudraCT, or similar)?")
    i03: Literal["YES", "NO", "UNSURE"] = Field(description="Is this an interventional study where participants receive a defined treatment, program, or intervention (as opposed to purely observational or qualitative research)?")
    i04: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants with schizophrenia or schizophrenia spectrum disorders?")
    i05: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants with non-affective psychotic disorders, such as first-episode psychosis, brief psychotic disorder, or delusional disorder?")
    i06: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants with schizoaffective disorder?")
    i07: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants with schizophreniform disorder?")
    i08: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants with schizotypy or schizotypal disorder?")
    i09: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants experiencing persistent hallucinations?")
    i10: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants at clinical or ultra-high risk for psychosis (e.g., prodromal psychosis, at-risk mental state)?")
    i11: Literal["YES", "NO", "UNSURE"] = Field(description="Is the study focused on participants with akathisia (as the primary condition or target population)?")
    i12: Literal["YES", "NO", "UNSURE"] = Field(description="Is the study focused on participants with tardive dyskinesia?")
    i13: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants diagnosed with serious mental illness (SMI) that includes psychotic disorders?")
    i14: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include participants with dual diagnosis (a psychotic disorder plus a substance use disorder)?")
    i15: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include relatives or caregivers of individuals with schizophrenia-spectrum disorders?")
    i16: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study focus on stigma, education, or awareness about schizophrenia or serious mental illness, even if the participants are from the general population?")
    i17: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study involve brain stimulation techniques (e.g., ECT, rTMS, tDCS, VNS, MST) targeting schizophrenia-spectrum populations?")

    e01: Literal["YES", "NO", "UNSURE"] = Field(description="Is this an animal study conducted exclusively on animals (no human participants)?")
    e02: Literal["YES", "NO", "UNSURE"] = Field(description="Is this a systematic review, meta-analysis, scoping review, narrative review, or literature review?")
    e03: Literal["YES", "NO", "UNSURE"] = Field(description="Is this a case report or case series (explicitly identified as such, or describing single-case design)?")
    e04: Literal["YES", "NO", "UNSURE"] = Field(description="Is this a retraction notice without substantive study content?")
    e05: Literal["YES", "NO", "UNSURE"] = Field(description="Is this a single-arm or single-group observational study with no control group?")
    e06: Literal["YES", "NO", "UNSURE"] = Field(description="Is this a purely observational, registry, or retrospective study without intervention assignment?")
    e07: Literal["YES", "NO", "UNSURE"] = Field(description="Is this a purely qualitative study (based on interviews, focus groups, or narratives) without any assigned intervention?")
    e08: Literal["YES", "NO", "UNSURE"] = Field(description="Is this a study conducted exclusively on healthy volunteers, such as physiological or Phase I studies?")
    e09: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study focus exclusively on affective psychosis, such as bipolar disorder, psychotic depression, or postpartum psychosis, without including non-affective psychotic disorders?")
    e10: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study involve brain stimulation targeting only mood disorders, and not psychotic disorders?")
    e11: Literal["YES", "NO", "UNSURE"] = Field(description="Does the study include relatives or caregivers supporting non-psychotic conditions, such as dementia, depression, or autism?")
    e12: Literal["YES", "NO", "UNSURE"] = Field(description="Is the document a purely administrative retraction notice without any substantive study details?")

def convert_yes_no_unsure_to_int(value: str, is_include: bool) -> int:
    """Convert Yes/No/Unsure to binary: NO->0, YES->1, UNSURE->1 or 0.5"""
    if value == "NO":
        return 0
    elif value == "YES":
        return 1
    elif value == "UNSURE":
        return 1 if is_include else 0
    else:
        raise ValueError(f"Unexpected value: {value}")
    
def extractions_to_binary(data):
    #features = ['E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'I1', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9']
    #vector = []
    #for feature in features:
    #    vector.append(convert_yes_no_unsure_to_int(data[feature], True))
    #return vector
    for key, value in data.items():
        data[key] = [convert_yes_no_unsure_to_int(value, key.upper().startswith("I"))]
    return data

def _build_cache_key(
    paper_title: str,
    paper_abstract: str,
    question_text: str,
    model_name: str = OPENAI_MODEL,
) -> str:
    payload = json.dumps(
        {
            "title": paper_title or "",
            "abstract": paper_abstract or "",
            "question": question_text or "",
            "model": model_name,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{REDIS_CACHE_PREFIX}:{digest}"


async def _get_cached_answer(cache_key: str):
    if not redis_client:
        return None

    try:
        cached_value = await redis_client.get(cache_key)
    except Exception as exc:
        print(f"Redis read failed for key {cache_key}: {exc}")
        return None

    if not cached_value:
        return None

    try:
        data = json.loads(cached_value)
        return data
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"Redis payload decode failed for key {cache_key}: {exc}")
        return None


async def _set_cached_answer(cache_key: str, answer: StructuredAnswer, prompt_tokens: int, completion_tokens: int):
    if not redis_client:
        return

    payload = json.dumps(
        {
            "answer": answer.model_dump(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )

    try:
        await redis_client.set(cache_key, payload, ex=REDIS_TTL_SECONDS)
    except Exception as exc:
        print(f"Redis write failed for key {cache_key}: {exc}")

async def check_exclusion_criteria(paper_title, paper_abstract, extraction_reasons=[]):
    """Call OpenAI Responses API for a screening question with Redis caching."""

    system_prompt = (
        "You are a high-recall screening assistant and an expert in title and abstract screening."
        "Your task is to evaluate a paper for possible exclusion criteria based on its title and abstract. "
        #"Because the goal is high recall, only return an exclusion decision if you are very certain (We don't want to miss relevant papers)."

        "The possible exclusion reasons are provided as a list of questions. Select the appropriate reason if the question evaluates to TRUE."
        "If multiple reasons apply, return multiple ids."
    )

    extraction_reasons = [str(i) + ": " + reason for i, reason in enumerate(extraction_reasons)]
    extraction_list = '\n'.join(extraction_reasons)

    user_prompt = (
        "This is the paper you are asked to evaluate:\n\n"
        f"Title: {paper_title}\n\n"
        f"Abstract: {paper_abstract}\n\n"
        "Task: Given this information and the following possible exclusion reasons, determine whether the paper should be excluded. "
        "If so, return the corresponding reason ids.\n\n"
        f"{extraction_list}"
    )

    try:
        response = await OPENAI_CLIENT.responses.parse(
            model=OPENAI_MODEL,
            reasoning={"effort": "low"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=ExclusionReason,
            store=True
        )

        result = response.output_parsed

        return result

    except Exception as e:
        print(f"Error processing exclusion criteria for paper {paper_title}: {e}")
        return None


async def call_llm_single_question(paper_title, paper_abstract, question_text, danger_mode=False):
    """Call OpenAI Responses API for a screening question with Redis caching."""

    cache_key = _build_cache_key(paper_title, paper_abstract, question_text, OPENAI_MODEL)
    cached_response = await _get_cached_answer(cache_key)
    if cached_response:
        return cached_response

    system_prompt = (
        "You are a high-recall screening assistant. You are an expert for title abstract screening."
        "Your task is to extract binary inclusion and exclusion criteria features."
        "Each inclusion/exclusion feature is formulated as a question."
        "The extracted binary features are then passed to a decision tree."
        "Aim for high recall, since we dont want to miss a single relevant paper."
        "You will always only see one question. Given this question start generating keywords you would expect in related study."
        "Respond with ONLY one word: YES, NO, or UNSURE "
        "If you don't have enough information respond with UNSURE\n"
    )

    if danger_mode:
        system_prompt += (
            "This task is highly critical: if you respond with YES, this paper will be rejected. "
            "To maintain high recall, carefully consider before answering YES."
        )

    user_prompt = (
        "This is the paper you are asked to evaluate:"
        f"Title: {paper_title}\n\n"
        f"Abstract: {paper_abstract}\n\n"
        "Task: Given this information, answer all questions."
        "Return only YES, NO, or UNSURE.\n\n"
        f"Please extract the following feature: {question_text}"
    )

    try:
        response = await OPENAI_CLIENT.responses.parse(
            model=OPENAI_MODEL,
            reasoning={"effort": "low"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=StructuredAnswer,
            store=True
        )


        answer = response.output_parsed

        await _set_cached_answer(
            cache_key,
            answer,
            getattr(response.usage, "input_tokens", 0),
            getattr(response.usage, "output_tokens", 0),
        )

        result = await _get_cached_answer(cache_key)

        return result

    except Exception as e:
        print(f"Error processing paper {paper_title}, question {question_text}: {e}")
        return None
    
async def call_llm_all_at_once(paper_title, paper_abstract):
    messages = [
        {
            "role": "system",
            "content": "You are a systematic review screening assistant. Your task is to determine if a paper meets specific inclusion or exclusion criteria based on its title and abstract. You must answer with ONLY one word: YES, NO, or UNSURE (only if you are really uncertain). Do not provide any explanation or additional text.",
        },
        {"role": "user", "content": f"Title: {paper_title}\n\nAbstract: {paper_abstract}"},
        {
            "role": "user",
            "content": f"Task: according to the paper's title and abstract, answer all following questions inclusion criteria (i01, i02, ...) and exclusion criteria (e01, e02, ...). Only answer with YES, NO or UNSURE (just if you are really unsure). Eventually return the structured response.",
        },
    ]

    try:
        response = await OPENAI_CLIENT.chat.completions.parse(model=OPENAI_MODEL, messages=messages, response_format=StructuredMultiAnswer)
        return response
    except Exception as e:
        print(f"Error processing paper {paper_title}: {e}")
        return None

async def extract_features(title, abstract, all_at_once=False):
    """
    Process all questions for a single paper sequentially.
    This preserves prompt caching for the paper's title+abstract.

    Returns: (paper_id, num_processed, num_skipped)
    """

    data = {}

    if all_at_once:
        response = await call_llm_all_at_once(title, abstract)
        total_input_tokens_used = response.usage.prompt_tokens
        total_output_tokens_used = response.usage.completion_tokens

        parsed_answers = response.choices[0].message.parsed
        for feature in ['E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'I1', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9']:
            key = feature[0].lower() + feature[1:].zfill(2)
            value = getattr(parsed_answers, key)
            data[feature] = value

    else:

        async def call_for_question(question_row):
            question_id = question_row["id"]
            question_text = question_row["question"]
            response, _ = await call_llm_single_question(title, abstract, question_text)
            return question_id, response

        tasks = [call_for_question(row) for _, row in LLM_QUESTIONS.iterrows()]
        results = await asyncio.gather(*tasks)
        total_input_tokens_used = 0
        total_output_tokens_used = 0
        for question_id, response in results:
            if response is None:
                raise RuntimeError(
                    f"LLM response missing for question {question_id}; aborting extraction"
                )
            total_input_tokens_used += response.usage.prompt_tokens
            total_output_tokens_used += response.usage.completion_tokens
            result = response.choices[0].message.parsed.answer
            #result = response.choices[0].message.content
            data[question_id] = result

    data = extractions_to_binary(data)

    return data, total_input_tokens_used, total_output_tokens_used