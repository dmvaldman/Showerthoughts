# Downloads the Project Euler problems/solutions into a jsonl file

import requests
import os
import json

def get_problem(num):
    url = f"https://projecteuler.net/minimal={num}"
    r = requests.get(url)
    return r.text

def check_problem(num, answer):
    url = f"https://projecteuler.net/problem={num}"
    r = requests.post(url, data={'guess': answer})
    # Check for string "Correct!" in r.text
    return "incorrect" not in r.text

def get_solutions():
    solutions = {}

    # walk code/project_euler/solutions/
    solution_dir = "code/project_euler/solutions/"
    for root, dirs, files in os.walk(solution_dir):
        # files in the form of pNNN.py where NNN is the number of the problem
        for file in files:
            if file.startswith('p') and file.endswith(".py"):
                num = int(file[1:4])
                with open(os.path.join(root, file), 'r') as f:
                    code = f.read()
                solutions[num] = code
    return solutions

def get_answers():
    # open code/project_euler/answers.txt
    answers = {}
    with open("code/project_euler/answers.txt", 'r') as f:
        for line in f:
            # line of the form Problem NNN: answer
            num, answer = line.split(':')
            num = int(num[7:])
            answer = answer.strip()
            answers[num] = answer
    return answers

if __name__ == "__main__":
    save_path = "datasets/code/project_euler.jsonl"

    # build dataset {'problem', 'number', 'solution'}
    solutions = get_solutions()
    answers = get_answers()

    for key, solution in solutions.items():
        problem_num = int(key)
        problem = get_problem(problem_num)

        if 'Right click' in problem:
            continue

        entry = {
            'number': problem_num,
            'problem': problem,
            'solution': solution,
            'answer': answers[problem_num]
        }

        # write to jsonl file
        with open(save_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')