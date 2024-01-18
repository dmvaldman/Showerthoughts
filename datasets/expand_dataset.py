import json
import dotenv
from utils import get_model_path, get_client
from tenacity import retry, stop_after_attempt

dotenv.load_dotenv()

# Options: ["openai", "perplexity", "together"]
provider = 'together'
client = get_client(provider)
client_verifier = get_client("openai")

default_system_prompt = "You are an insightful and intuitive mathematician. You're great at solving math problems via brainstorming and non-linear reasoning. Sometimes you go down dead-ends but eventually you uncover and synthesize the insights needed to form the correct answer."

def run_conversation(
        prompts,
        model,
        system_prompt=default_system_prompt,
        client=client,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format=None):

    messages=[
        {"role": "system", "content": system_prompt}
    ]

    for prompt in prompts:
        message = {"role": "user", "content": prompt}
        messages.append(message)
        print(f'\n\nAssistant:\n\n{prompt}')

        reply = run_conversation_with_model(messages, client=client, model=model, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, response_format=response_format)
        new_message = {"role": "assistant", "content": reply}
        print(f'\n\nUser:\n\n{reply}')

        yield new_message

        messages.append(new_message)

    return messages

# annotate to try several times if it fails
@retry(stop=stop_after_attempt(3))
def call_llm(messages, client=client, model=None, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, response_format=None):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format
    )
    return response

def run_conversation_with_model(messages, client=None, model=None, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, response_format=None):
    finished = False
    reply = ''
    while not finished:
        response = call_llm(
            messages,
            client=client,
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format)
        finish_reason = response.choices[0].finish_reason

        if finish_reason == 'length':
            raise Exception("Conversation too long")
        elif finish_reason == "content_filter":
            raise Exception("Conversation filtered")

        # finish reason not implemented in some providers
        finished = (finish_reason == "stop") or (finish_reason is None)
        reply += response.choices[0].message.content
    return reply

def compare_solutions(problem, solution1, solution2, model="gpt-4-1106-preview"):
    prompts = [
        f"""Below is a problem from the William Lowell Putnam competition and a solution written by two different people.\n\nProblem:\n{problem}\n\nSolution 1:\n{solution1}\n\nSolution 2:\n{solution2}\n\nCompare the two solutions. Describe the similarities and differences between the two solutions.""",
        "If both solutions are correct and solve the problem, write `BOTH`. If the first solution solves the problem and the second doesn't, write `FIRST`. If the second solution solves the problem and the first doesn't, write `SECOND`. Don't write anything else."
    ]
    messages = []
    for message in run_conversation(prompts, client=client_verifier, model=model):
        messages.append(message)

    last_message = messages[-1]['content']
    return last_message

def parse_messages(messages):
    parsed_messages = []
    for message in messages:
        if message['role'] == 'assistant':
            parsed_message = message['content']
            parsed_messages.append(parsed_message)
    return parsed_messages

def get_conversation_prompts(problem, insights=None):
    first_prompt = f"Below is a problem from the William Lowell Putnam competition and its solution.\n\nProblem:\n{problem}\n\nIf you're stuck, here are the key insights to solving the problem\n\n{insights}"

    prompts = [
        first_prompt + "\n\n" + "Write out a fictional scenario of trying (and failing) to solve the problem because you didn't grasp the first key insight. Detail a few false starts that lead you to realize the first critical insight. Show, don't tell. Every step should be justified. The conversation should end before making any progress towards the second key insight (we'll do that later). Don't break character (speak from the first person, describe how you are solving the problem, and don't mention anything about the instructions). Some first steps to try could be: write out a few simple examples to get a feel for the problem, remind yourself of some relevant theorems or related problems, remind yourself of various proof/problem solving strategies, etc (these are just a few suggestions, only do what you think is appropriate for the problem at hand).",
        "Continuing this narrative, let's fail to realize the second key insight but then stumble our way to it once we're stuck like before. Again, don't break character and don't mention the instructions (such as referring to 'key insights'). End the conversation before making any progress towards the next insight.",
        "Continue this way by getting stuck on failing to grasp the remaining key insights, getting stuck, and then grasping them, until the successful conclusion of the proof. Again, don't break character and don't mention the instructions (such as referring to 'key insights').",
        "Now that we have the main ideas, let's clean up the proof and write our answer in formal and concise mathematics."
    ]

    return prompts

def find_steps_with_error(problem, solution, steps, model=None):
    steps_str = ''
    for index, step in enumerate(steps):
        steps_str += f"Step {index}:\n{step}\n\n"
    steps_str = steps_str.strip()

    prompt = f"""Below is a problem from the William Lowell Putnam competition and its solution.

Problem:
{problem}

Solution:
{solution}

A student is attempting to solve a Putnam problem. They are reasnoning aloud. Your job is to rate each of these reasoning steps. The steps may meander or go down the wrong direction. This is fine as long as the student either admits they're wrong/stuck or corrects themselves eventually.
Ultimately we are assessing the student's ability to reason through the problem and draw from relevant knowledge.

A positive step is one that advances towards the solution (this is given a result of 1).
A neutral step (0) is one that makes no progress towards the solution (this is given a result of 0).
A negative step (-1) is one that is logically incorrect or doesn't follow from the previous steps (this is given a result of -1).

Steps:

{steps_str}

Output your results in JSON with the following schema `{{steps: [{{step: step_num, explanation: explanation, result: result}},...]}}` where `step_num` is an integer, `explanation` is a brief explanation as a string, and `result` is an integer (-1, 0, or 1) indicating the rating of the step as above."""

    system_prompt = 'You are a mathematician and former Putnam gold medalist. You respond only in JSON.'
    messages = []
    for messages in run_conversation([prompt], system_prompt=system_prompt, client=client_verifier, model=model, response_format={"type": "json_object"}):
        messages.append(messages)

    return json.loads(messages[-1]['content'])

def get_insights(problem, solution, model_verifier):
    prompt = f"Below is a problem from the William Lowell Putnam competition and its solution.\n\nProblem:\n{problem}\n\nSolution:\n{solution}\n\nList the non-obvious key insights of the solution (choose up to 3 of the most non-obvious ones). A key insight is something non-obvious that if you knew ahead of time it would make solving the problem much more straightforward."
    messages = []
    for message in run_conversation([prompt], client=client_verifier, model=model_verifier):
        messages.append(message)

    insights = messages[-1]['content']
    return insights

def get_steps_and_solution(problem, model, system_prompt=default_system_prompt, insights=None):
    prompts = get_conversation_prompts(problem, insights=insights)

    messages = []
    for message in run_conversation(prompts, model=model, system_prompt=system_prompt):
        messages.append(message)

    steps = parse_messages(messages)
    generated_solution = steps[-1]
    return steps[0:-1], generated_solution

def add_to_dataset(dataset, steps, steps_annotated, data, generated_solution, model, model_verifier, is_solution_correct):
    # process steps
    processed_steps = []
    for step, step_annotated in zip(steps, steps_annotated):
        if step_annotated['result'] == 'WRONG':
            result = -1
        elif step_annotated['result'] == 'OKAY':
            result = 1
        else:
            result = 0

        processed_steps.append({
            'step': step,
            'result': result,
            'explanation': step_annotated['explanation']
        })

    dataset.append({
        'year': data['year'],
        'label': data['label'],
        'id': f"{data['year']}_{data['label']}",
        'problem': data['problem'],
        'solution': data['solution'],
        "model": model,
        "model_verifier": model_verifier,
        'generated_solution': generated_solution,
        'is_solution_correct': is_solution_correct,
        'steps': processed_steps
    })

    return dataset

def main(dataset, model, model_verifier, save=True):
    step_dataset = []
    for index, data in enumerate(dataset):
        if index > 10:
            break

        problem = data['problem']
        solution = data['solution']

        insights = get_insights(problem, solution, model_verifier)
        steps, generated_solution = get_steps_and_solution(problem, model=model, insights=insights)

        # print('\n\n\n\n')
        choice = compare_solutions(data['problem'], solution, generated_solution, model=model_verifier)
        is_solution_correct = (choice == 'SECOND' or choice == 'BOTH')
        # print('\n\n\n\n')

        # find steps with error
        steps += ['Answer:\n' + generated_solution]
        steps_annotated = find_steps_with_error(problem, solution, steps, model=model_verifier)

        # add to dataset
        step_dataset = add_to_dataset(step_dataset, steps, steps_annotated['steps'], data, generated_solution, model, model_verifier, is_solution_correct)

        # save dataset
        if save:
            model_name = model.replace('/', '_')
            save_name = f'datasets/putnam_steps_{model_name}.json'
            with open(save_name, 'w') as f:
                json.dump(step_dataset, f, indent=4)


if __name__ == "__main__":
    # Options: ["gpt-4-1106-preview", "gpt-3.5-turbo", "mistral-7b-instruct", "mixtral-8x7b-instruct", "llama2-70b-chat", "code-llama-34b-instruct", "llama2-13b-chat", "llama2-70b-instruct", "wizardlm-70b", "wizardlm-13b", "llema-7b"]
    model_nickname = "mistral-7b-instruct"
    # model_nickname = "mixtral-8x7b-instruct"
    model = get_model_path(model_nickname, provider)
    model_verifier = "gpt-4-1106-preview"

    # load dataset
    with open('datasets/putnam.json') as f:
        dataset = json.load(f)

    main(dataset, model, model_verifier, save=True)
