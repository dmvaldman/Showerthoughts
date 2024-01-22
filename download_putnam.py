import requests
import re
import json

def download_year(year):
    url_problem = f"https://kskedlaya.org/putnam-archive/{year}.tex"
    url_solution = f"https://kskedlaya.org/putnam-archive/{year}s.tex"

    headers = {
        'Accept': '*/*',  # This indicates that any response type is acceptable
        'Accept-Charset': 'UTF-8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # download file located at url_problem with content type text/plain
    response_problem = requests.get(url_problem, allow_redirects=True, headers=headers)
    response_solution = requests.get(url_solution, allow_redirects=True, headers=headers)

    if response_problem.status_code == 200 and response_solution.status_code == 200:
        problems = response_problem.text
        solutions = response_solution.text

        if response_problem.apparent_encoding == 'shift_jis_2004':
            problems = problems.replace('¥', '\\')

        if response_solution.apparent_encoding == 'shift_jis_2004':
            solutions = solutions.replace('¥', '\\')

        return problems, solutions
    else:
        return None, None

def parse_tex(tex):
    # regex for text inbetween \item[A-*], \item[A--*], \item[B-*], \item[B--*]
    # groups are (A|B), number, and text
    # pattern = r'\\item\[(A|B)-?-?(\d+)\](.+?)(?=\\item\[|$)'
    pattern = r'\\item\[(A|B)-{0,2}(\d+)\](.+?)(?=\\item\[|$)'
    matches = re.findall(pattern, tex, re.DOTALL)

    # trim all the text
    matches = [(letter, level, text.strip()) for letter, level, text in matches]
    return matches


# build dataset
dataset = []
year_range = range(1995, 2024)
for year in year_range:
    problems, solutions = download_year(year)
    problems = parse_tex(problems)
    solutions = parse_tex(solutions)

    assert len(problems) == len(solutions)

    for problem, solution in zip(problems, solutions):
        dataset.append({
            'year': year,
            'label': problem[0] + problem[1],
            'difficulty': problem[1],
            'problem': problem[2],
            'solution': solution[2]
        })

# save dataset
with open('datasets/putnam/putnam.json', 'w') as f:
    json.dump(dataset, f, indent=2)