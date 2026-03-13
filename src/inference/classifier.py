from joblib import load
from .llm_extractor import call_llm_single_question, check_exclusion_criteria
import numpy as np
import unicodedata
from bs4 import BeautifulSoup

QUESTIONS = {
    #'I01' : "Does the paper describe a randomized controlled trial (RCT), controlled clinical trial (CCT), or any trial with prospective assignment of participants?",
    'I01' : "Does the paper describe a randomized controlled trial (RCT), controlled clinical trial (CCT), or any trial with prospective assignment of participants? If a trial registration identifier (e.g., NCT, ISRCTN, EudraCT, or similar) is given, it is most likely a RCT.",
    'I02' : "Does the paper include or reference a trial registry identifier (e.g., NCT, ISRCTN, EudraCT, or similar)?",
    'I03' : "Is this an interventional study where participants receive a defined treatment, program, or intervention (as opposed to purely observational or qualitative research)?",
    #'I04' : "Does the study include participants with schizophrenia or schizophrenia spectrum disorders?",
    'I04' : "Does the study explicitly include participants with schizophrenia or schizophrenia spectrum disorders?",
    'I05' : "Does the study include participants with non-affective psychotic disorders, such as first-episode psychosis, brief psychotic disorder, or delusional disorder?",
    'I06' : "Does the study include participants with schizoaffective disorder?",
    'I07' : "Does the study include participants with schizophreniform disorder?",
    'I08' : "Does the study include participants with schizotypy or schizotypal disorder?",
    'I09' : "Does the study include participants experiencing persistent hallucinations?",
    'I10' : "Does the study include participants at clinical or ultra-high risk for psychosis (e.g., prodromal psychosis, at-risk mental state)?",
    'I11' : "Does the study focus on participants with akathisia (as the primary condition or target population)?",
    'I12' : "Does the study focus on participants with tardive dyskinesia?",
    'I13' : "Does the study include participants diagnosed with serious mental illness (SMI) that includes psychotic disorders?",
    'I14' : "Does the study include participants with dual diagnosis (a psychotic disorder plus a substance use disorder)?",
    'I15' : "Does the study include relatives or caregivers of individuals with schizophrenia-spectrum disorders?",
    'I16' : "Does the study focus on stigma, education, or awareness about schizophrenia or serious mental illness, even if the participants are from the general population?",
    'I17' : "Does the study involve brain stimulation techniques (e.g., ECT, rTMS, tDCS, VNS, MST) targeting schizophrenia-spectrum populations?",

    'E01' : "Is this an animal study conducted exclusively on animals (no human participants)?",
    'E02' : "Is this a systematic review, meta-analysis, scoping review, narrative review, or literature review?",
    'E03' : "Is this a case report or case series (explicitly identified as such, or describing single-case design)?",
    'E04' : "Is this a retraction notice without substantive study content?",
    'E05' : "Is this a single-arm or single-group observational study with no control group?",
    'E06' : "Is this a purely observational, registry, or retrospective study without intervention assignment?",
    'E07' : "Is this a study based entirely on qualitative methods (based on interviews, focus groups, or narratives) without any assigned intervention?",
    'E08' : "Is this a study conducted exclusively on healthy volunteers, such as physiological or Phase I studies?",
    'E09' : "Does the study focus exclusively on affective psychosis, such as bipolar disorder, psychotic depression, or postpartum psychosis, without including non-affective psychotic disorders?",
    'E10' : "Does the study involve brain stimulation targeting only mood disorders, and not psychotic disorders?",
    'E11' : "Does the study include relatives or caregivers supporting non-psychotic conditions, such as dementia, depression, or autism?",
    'E12' : "Is the document a purely administrative retraction notice without any substantive study details?",
}

EXCLUSION_REASONS = [
    "Is this an animal study conducted exclusively on animals (no human participants)?",
    "Is this a systematic review, meta-analysis, scoping review, narrative review, or literature review?",
    "Is this a case report or case series (explicitly identified as such, or describing single-case design)?",
    #"Is this a retraction notice without substantive study content?",
    #"Is this a single-arm or single-group observational study with no control group?",
    #"Is this a purely observational, registry, or retrospective study without intervention assignment?",
    "Is this a purely observational, registry-based, or retrospective study with no intervention assignment, and does the observational analysis not stand from a randomized controlled trial (no rct mentioned in the abstract)?", #unless the observational study belongs to a randomized 
    #"Is this a study based entirely on qualitative methods (based on interviews, focus groups, or narratives) without any assigned intervention?",
    "Is this a study based ENTIRELY on qualitative methods (based on interviews, focus groups, or narratives) without any assigned intervention? Also no trial is mentioned.",
    "Is this a study conducted exclusively on healthy volunteers, such as physiological or Phase I studies?",
    "Does the study focus exclusively on affective psychosis, such as bipolar disorder, psychotic depression, or postpartum psychosis, without including non-affective psychotic disorders?",
    "Does the study involve brain stimulation targeting only mood disorders, and not psychotic disorders?",
    "Does the study include relatives or caregivers supporting non-psychotic conditions, such as dementia, depression, or autism?",
    #"Is the document a purely administrative retraction notice without any substantive study details?",
]

FEATURE_ORDER = ['E01', 'E10', 'E11', 'E12', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09',
                 'I01', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I02', 'I03', 'I04',
                 'I05', 'I06', 'I07', 'I08', 'I09']

FEATURE_ORDER = ['I01', 'I10', 'I13', 'I16', 'I17', 'I04', 'I05', 'I07']

NUMBER_OF_EXCLUSION_CHECKS = 1

MODEL = load("dt_smote_pipeline_model.joblib")

async def traverse_tree(tree, title, abstract, node=0, all_answers=None):
    tree_ = tree.tree_
    # Base case: leaf node
    if tree_.children_left[node] == tree_.children_right[node]:
        return tree_.value[node], all_answers
    
    feature_index = tree_.feature[node]
    #threshold = tree_.threshold[node]
    current_feature_name = FEATURE_ORDER[feature_index]

    # Track unique feature extraction
    #if current_feature_name not in extracted_features:
    current_feature_question = QUESTIONS[current_feature_name]
    llm_extraction = await call_llm_single_question(title, abstract, current_feature_question)
    llm_prediction = llm_extraction['answer']['answer']
    llm_evidence = llm_extraction['answer']['evidence']
    
    #current_feature_value = extracted_features[current_feature_name]
    left_child = tree_.children_left[node]
    right_child = tree_.children_right[node]

    left_values = tree_.value[left_child][0]
    right_values = tree_.value[right_child][0]

    left_prob = left_values[1] / np.sum(left_values)
    right_prob = right_values[1] / np.sum(right_values)

    next_node = -1
    
    if llm_prediction == "NO":
        next_node = left_child
    elif llm_prediction == "YES":
        next_node = right_child
    elif llm_prediction == "UNSURE":

        if left_prob > right_prob:
            next_node = left_child
        else:
            next_node = right_child

    score = left_prob if next_node == left_child else right_prob

    all_answers.append({'question': current_feature_question, 'answer': llm_prediction, 'evidence': llm_evidence, 'inclusion_score': score})
        
    return await traverse_tree(tree, title, abstract, next_node, all_answers=all_answers)

async def predict_proba(title: str, abstract: str) -> float:
    # This dictionary stores: { "E01": 1, "I04": 0, ... }
    # It prevents redundant LLM calls and serves as our counter
    #extracted_features = {}
    
    """
    all_proba = np.array([0.0, 0.0])
    for tree in MODEL.estimators_:
        current_proba = await traverse_tree(tree, title, abstract, extracted_features)
        all_proba += current_proba[0] # Estimator values are usually wrapped in an extra array

    all_proba /= len(MODEL.estimators_)
    """

    title_soup = BeautifulSoup(title, "lxml")
    title = title_soup.get_text()

    abstract_soup = BeautifulSoup(abstract, "lxml")
    abstract = abstract_soup.get_text()

    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8')
    abstract = unicodedata.normalize('NFKD', abstract).encode('ascii', 'ignore').decode('utf-8')

    exclusion_reasons_result = await check_exclusion_criteria(title, abstract, EXCLUSION_REASONS)
    for index in exclusion_reasons_result.ids:
        #TODO run multiple checks in parallel NUMBER_OF_EXCLUSION_CHECKS
        llm_extraction = await call_llm_single_question(title, abstract, EXCLUSION_REASONS[index], danger_mode=True)
        llm_prediction = llm_extraction['answer']['answer']
        if llm_prediction == "YES":
            llm_evidence = llm_extraction['answer']['evidence']
            return 0, [{'question': EXCLUSION_REASONS[index], 'answer':llm_prediction, 'evidence' : llm_evidence, 'score': 0}],  {'title': title, 'abstract': abstract}

    all_proba, extracted_features = await traverse_tree(MODEL, title, abstract,0,[])
    
    # Print the total count
    return all_proba[0][1], extracted_features, {'title': title, 'abstract': abstract}
