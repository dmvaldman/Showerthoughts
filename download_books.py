import requests
import json
import dotenv
import os
import PyPDF2
import time
import re

dotenv.load_dotenv()

headers_mathpix = {
    "app_id": os.getenv("MATHPIX_APP_ID"),
    "app_key": os.getenv("MATHPIX_API_KEY")
}

# 500 mathematical challenges.pdf
# problems start on page 14-59. annotated by "Problem #N. Text of problem"
# solutions start on page 60-222. annotated by "Problem #N. Solution."

def convert_page_range_to_string(page_range):
    # page range provided in ranges and single pages
    # page range given by [(start, end), (start, end), start, ....]
    page_ranges_str = ''
    for page in page_range:
        if isinstance(page, tuple):
            page_ranges_str += f"{page[0]}-{page[1]}_"
        else:
            page_ranges_str += f"{page}_"
    page_ranges_str = page_ranges_str[:-1]  # remove last character
    return page_ranges_str


def trim_pdf(path, page_ranges):
    page_ranges_str = convert_page_range_to_string(page_ranges)
    output_path = path.replace(".pdf", f"_trim_{page_ranges_str}.pdf")

    # if path exists, warn
    if os.path.exists(output_path):
        print(f"Warning: Trimmed PDF at {output_path} already exists. Skipping.")
        return output_path

    pdf_writer = PyPDF2.PdfWriter()

    with open(path, 'rb') as infile:
        pdf_reader = PyPDF2.PdfReader(infile)

        for page_range in page_ranges:
            if isinstance(page_range, tuple):
                start, end = page_range
                for page_number in range(start-1, end):  # PyPDF2 uses 0-based index
                    pdf_writer.add_page(pdf_reader.pages[page_number])
            elif isinstance(page_range, int):
                start = page_range
                pdf_writer.add_page(pdf_reader.pages[start-1])

        with open(output_path, 'wb') as outfile:
            pdf_writer.write(outfile)

    return output_path

def upload_pdf(path, force=False):
    with open('books/pdf_ids.json', 'r') as f:
        pdf_ids = json.load(f)

    if not force and path in pdf_ids:
        print(f"PDF at {path} already uploaded. Skipping.")
        return pdf_ids[path]

    r = requests.post("https://api.mathpix.com/v3/pdf",
        json={
            "conversion_formats": {"docx": True, "tex.zip": True},
            "remove_section_numbering": True
        },
        headers=headers_mathpix,
        files={
            "file": open(path, "rb")
        }
    )

    if r.status_code != 200:
        raise Exception(f"Error uploading PDF at {path}.\n{r}.")

    pdf_id = r.json()['pdf_id']

    # save pdf id
    pdf_ids[path] = pdf_id
    with open('books/pdf_ids.json', 'w') as f:
        json.dump(pdf_ids, f, indent=4)

    return pdf_id

def process_pdf(pdf_id, save_path=None):
    if os.path.exists(save_path):
        print(f"Warning: Processed PDF at {save_path} already exists. Skipping.")
        return

    completed = False
    while not completed:
        # get request in json
        r = requests.get(f"https://api.mathpix.com/v3/pdf/{pdf_id}", headers=headers_mathpix)
        response = r.json()
        if 'error' in response:
            raise Exception(f"Error processing PDF at {pdf_id}.\n{response['error']}")
        completed = (response["status"] == 'completed')
        if not completed:
            time.sleep(1)

    # get content
    filetype = ".mmd"
    r = requests.get(f"https://api.mathpix.com/v3/pdf/{pdf_id}" + filetype, headers=headers_mathpix)

    r.text.encode('utf-8')
    with open(save_path, "w", encoding='utf-8') as f:
        f.write(r.text)

def pdf2mmd(pdf_path, page_ranges):
    pdf_path_trimmed = trim_pdf(pdf_path, page_ranges)
    pdf_id = upload_pdf(pdf_path_trimmed, force=False)
    path_processed = pdf_path_trimmed.replace(".pdf", ".mmd")
    process_pdf(pdf_id, save_path=path_processed)
    return path_processed

def process_500_mathematical_challenges(path, save=False):
    path_save = path.replace(".mmd", ".jsonl")

    if os.path.exists(path_save):
        print(f"Warning: Processed PDF at {path_save} already exists. Skipping.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    problems_with_numbers = re.findall(r'Problem (\d+)\. (.*?)(?=\n\nProblem \d+|$)', text, re.DOTALL)
    problems_array = [(int(number), text.strip()) for number, text in problems_with_numbers]

    # loop through array, first occurences of an integer are the problems, second are the solutions
    problems = {}
    solutions = {}
    min_num = 100
    max_num = -100
    for number, text in problems_array:
        if number > max_num:
            problems[number] = text
        else:
            solutions[number] = text

        max_num = max(max_num, number)
        min_num = min(min_num, number)

    if save:
        # zip problems and solutions
        # clear file first
        open(path_save, 'w').close()
        with open(path_save, 'a', encoding='utf-8') as f:
            for number in range(1, max_num + 1):
                if number in problems and number in solutions:
                    record = {
                        'problem': problems[number],
                        'solution': solutions[number]
                    }
                    f.write(json.dumps(record) + '\n')

def process_USSR_olympiad(path, save=False):
    path_save = path.replace(".mmd", ".jsonl")

    if os.path.exists(path_save):
        print(f"Warning: Processed PDF at {path_save} already exists. Skipping.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Regular expression to match the problems
    # pattern = r'\n(\d+)\.\s(.*?)\n(?=\n\d+\.|\n*$)'
    pattern = r'\n(\d+\*?)\.?\s*(.*?)\n(?=\n\d+\*?\.|\Z)'

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)
    problems_array = [(int(num.rstrip('*')), re.sub(r'\s+', ' ', text.strip('*').strip())) for num, text in matches]

    # loop through array, first occurences of an integer are the problems, second are the solutions
    problems = {}
    solutions = {}
    hints_and_answers = {}
    prev_num = 0
    curr_pass = 1
    for number, text in problems_array:
        if number < prev_num:
            curr_pass += 1

        if curr_pass == 1:
            problems[number] = text
        elif curr_pass == 2:
            solutions[number] = text
        elif curr_pass == 3:
            hints_and_answers[number] = text

        prev_num = number

    if save:
        # zip problems and solutions
        # clear file first
        open(path_save, 'w').close()
        with open(path_save, 'a', encoding='utf-8') as f:
            for number in problems.keys():
                if number in solutions:
                    hint_or_answer = hints_and_answers[number] if number in hints_and_answers else None
                    record = {
                        'problem': problems[number],
                        'solution': solutions[number],
                        'hints_and_answers': hint_or_answer
                    }
                    f.write(json.dumps(record) + '\n')

def process_IMO_Compendium(path, save=False):
    path_save = path.replace(".mmd", ".jsonl")

    if os.path.exists(path_save):
        print(f"Warning: Processed PDF at {path_save} already exists. Skipping.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_problems, text_solutions = text.split('\section*{Solutions}')

    # Pattern to extract the section with its year and problems
    section_pattern = r'\\subsection\*\{[\d.]+ The .*?(\d{4})\}(.*?)(?=\\subsection\*\{[\d.]+ The |\Z)'

    # Find all section matches
    section_matches_problems = re.findall(section_pattern, text_problems, re.DOTALL)

    problems = {}
    for section in section_matches_problems:
        year = int(section[0])
        section_text = section[1]

        type = None
        if 'Content Problems' in section_text:
            type = 'Contest'
        elif 'Shortlisted Problems' in section_text:
            type = 'Shortlisted'
        elif 'Longlisted Problems' in section_text:
            type = 'Longlisted'

        # Extract problems within this section
        # This regex pattern matches:
        # - A problem number followed by a period
        # - Optionally, a country code in parentheses (with potential additional characters)
        # - The problem text
        problem_pattern = r'\n(\d+)\. (?:\([A-Z]{3}(?: \d)?\)\s*)?(.*?)(?=\n\d+\.|\Z)'
        problem_matches = re.findall(problem_pattern, section_text, re.DOTALL)

        if problem_matches:
            for problem in problem_matches:
                identifier = f"{year}_{type}_{problem[0]}"
                problem_text = problem[1].strip()
                problems[identifier] = problem_text
        else:
            print(f"No problems found for the year {year}")

        solution_section_pattern = r'\\subsection\*\{4\.\d+ Solutions to the (Contest|Shortlisted|Longlisted) Problems of IMO (\d{4})\}(.*?)(?=\\subsection\*\{|\\section\*|\Z)'
        solution_section_matches = re.findall(solution_section_pattern, text_solutions, re.DOTALL)

        solutions = {}
        for section in solution_section_matches:
            type = section[0]
            year = section[1]
            section_text = section[2]

            solution_pattern = r'\n(\d+)\. (?:\([A-Z]{3}(?: \d)?\)\s*)?(.*?)(?=\n\d+\.|\Z)'
            solution_matches = re.findall(solution_pattern, section_text, re.DOTALL)

            for solution in solution_matches:
                identifier = f"{year}_{type}_{solution[0]}"
                solution_text = solution[1].strip()
                solutions[identifier] = solution_text

    shared_keys = set(solutions.keys()) & set(problems.keys())

    if save:
        # zip problems and solutions
        # clear file first
        open(path_save, 'w').close()
        with open(path_save, 'a', encoding='utf-8') as f:
            for key in shared_keys:
                record = {
                    'problem': problems[key],
                    'solution': solutions[key]
                }
                f.write(json.dumps(record) + '\n')


# pdf_path = "books/500 mathematical challenges/original.pdf"
# page_ranges = [(14, 59), (60, 222)]
# path_processed = pdf2mmd(pdf_path, page_ranges)
# process_500_mathematical_challenges(path_processed, save=True)

# pdf_path = "books/The USSR Olympiad Problem Book/original.pdf"
# page_ranges = [(27, 100), (101, 473)]
# path_processed = pdf2mmd(pdf_path, page_ranges)
# process_USSR_olympiad(path_processed, save=True)

# The IMO Compendium
pdf_path = "books/The IMO Compendium/original.pdf"
page_ranges = [(21, 326), (327, 704)]
path_processed = pdf2mmd(pdf_path, page_ranges)
process_IMO_Compendium(path_processed, save=True)