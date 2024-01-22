import requests

def get_problem(num):
    url = f"https://projecteuler.net/minimal={num}"
    r = requests.get(url)
    return r.text

def check_problem(num, answer):
    url = f"https://projecteuler.net/problem={num}"
    r = requests.post(url, data={'guess': answer})
    # Check for string "Correct!" in r.text
    return "incorrect" not in r.text

if __name__ == "__main__":
    print(get_problem(1))
    result = check_problem(1, 233168)
    print('hi')