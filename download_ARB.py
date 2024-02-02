# From https://arxiv.org/abs/2307.13692
# ARB: Advanced Reasoning Benchmark for Large Language Models
# https://advanced-reasoning-benchmark.netlify.app/documentation

import requests
import json

def download_law():
    url = "https://advanced-reasoning-benchmark.netlify.app/api/lib/law/"
    response = requests.get(url)
    data = response.json()
    # keys: Problem Statement, Answer Candidates (array), Final Answer, Problem Type (multiple choice)
    # num questions: 627
    print('hi')

def download_math():
    answer_types = ["numerical", "symbolic", "prooflike"]
    for answer_type in answer_types:
        url = f"https://advanced-reasoning-benchmark.netlify.app/api/lib/math/{answer_type}/"
        response = requests.get(url)
        data = response.json()
        # keys: Problem_Statement, Topic, Output Format Instructions, Solution, Final Answer, Problem Type, rubric
        # num questions: 69, 52, 19
        print('hi')

def download_physics():
    answer_types = ["numerical", "symbolic"]
    for answer_type in answer_types:
        url = f"https://advanced-reasoning-benchmark.netlify.app/api/lib/physics/{answer_type}/noimg"
        response = requests.get(url)
        data = response.json()
        # keys: Problem_Statement, Topic, Output Format Instructions, Solution, Final Answer, Problem Type, rubric
        # num questions: 113, 51
        print('hi')

if __name__ == "__main__":
    download_law()
    download_math()
    download_physics()