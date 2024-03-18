# Generate synthetic non-linear reasoning steps inbetween problem/solution of dataset

import json
import dotenv
from utils import get_model_path, get_client
from tenacity import retry, stop_after_attempt
from datetime import datetime
import time

dotenv.load_dotenv()

MAX_TOKENS_CLAUDE = 4096

def parse_messages(messages):
    parsed_messages = []
    for message in messages:
        if message['role'] == 'assistant':
            parsed_message = message['content']
            parsed_messages.append(parsed_message)
    return parsed_messages

def response_to_finish_reason(client, response):
    if type(client).__name__ == "OpenAI":
        return response.choices[0].finish_reason
    elif type(client).__name__ == "Anthropic":
        return response.stop_reason

def response_to_text(model, response):
    if type(model).__name__ == "OpenAI":
        return response.choices[0].message.content.strip()
    elif type(model).__name__ == "Anthropic":
        return response.content[0].text.strip()

class DatasetExpander():
    system_prompt = "You are an insightful and intuitive mathematician. You're great at solving math problems via brainstorming and non-linear reasoning. Sometimes you go down dead-ends but eventually you uncover and synthesize the insights needed to form the correct answer."
    system_prompt_verifier = "You are a mathematician and former Putnam gold medalist."
    def __init__(self, dataset, provider, provider_verifier, model, model_verifier, max_attempts_generator=4, max_attempts_verifier=2):
        self.dataset = dataset
        self.provider = provider

        self.max_attempts_generator = max_attempts_generator
        self.max_attempts_verifier = max_attempts_verifier

        # generator model
        self.model = get_model_path(model, provider)
        self.client = get_client(provider)

        # verifier model
        self.model_verifier = get_model_path(model_verifier, provider_verifier)
        self.client_verifier = get_client(provider_verifier)

        self.dataset_steps = []

    def call_llm(self, system_prompt, messages, type, llm_options={}, verbose=False):
        if type == 'generator':
            model = self.model
            client = self.client
        elif type == 'verifier':
            model = self.model_verifier
            client = self.client_verifier

        finished = False
        reply = ''
        while not finished:
            response = self.call_llm_base(system_prompt, messages, model, client, llm_options, verbose=verbose)
            finish_reason = response_to_finish_reason(client, response)

            if finish_reason == 'length' or finish_reason == 'max_tokens':
                raise Exception("Conversation too long")
            elif finish_reason == "content_filter":
                raise Exception("Conversation filtered")

            # finish reason not implemented in some providers
            finished = finish_reason in ["stop", "eos", "end_turn", None]

            content = response_to_text(client, response)
            reply += content

        return reply

    @retry(stop=stop_after_attempt(3))
    def call_llm_base(self, system_prompt, messages, model, client, llm_options={}, verbose=False):
        json_mode = "response_format" in llm_options and llm_options["response_format"]["type"] == "json_object"
        client_name = type(client).__name__

        try:
            if client_name == "OpenAI":
                system_message = {"role": "system", "content": system_prompt}
                messages = [system_message] + messages
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **llm_options
                )

                # attempt parsing as JSON
                if json_mode:
                    try:
                        json.loads(response.choices[0].message.content)
                    except:
                        raise Exception("Response not valid JSON")

            elif client_name == "Anthropic":
                if json_mode:
                    # prefill with { to constrain to JSON
                    messages.append({"role": "assistant", "content": "{"})

                response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=messages,
                    max_tokens=MAX_TOKENS_CLAUDE
                )

                if json_mode:
                    # prepend with {
                    response.content[0].text = "{" + response.content[0].text
                return response
        except Exception as e:
            print("Couldn't call LLM, retrying...", e)
            raise e

        if verbose:
            # print conversation
            request_text = messages[-1]['content']
            response_text = response_to_text(client, response)
            print(f"User:\n{request_text}")
            print(f"Assistant:\n{response_text}")

        return response

    def run_conversation(self, prompts, system_prompt=None, type='generator', llm_options={}, verbose=False):
        if system_prompt is None:
            if type=='generator':
                system_prompt = DatasetExpander.system_prompt
            elif type=='verifier':
                system_prompt = DatasetExpander.system_prompt_verifier

        messages = []

        for prompt in prompts:
            message = {"role": "user", "content": prompt}
            messages.append(message)
            last_step = self.call_llm(system_prompt, messages, type=type, llm_options=llm_options, verbose=verbose)
            new_message = {"role": "assistant", "content": last_step}
            messages.append(new_message)

        return messages

    def run_conversation_with_checks(self, prompts, problem=None, solution=None, llm_options={}):
        system_prompt = DatasetExpander.system_prompt
        messages = []
        dataset_steps = []

        for prompt in prompts:
            message = {"role": "user", "content": prompt}
            messages.append(message)

            passed_check = False
            steps = parse_messages(messages)
            step_data = {
                "completions": [],
                "chosen_index": None,
                "model": None
            }

            type = 'generator'
            index = 0

            while not passed_check and len(step_data['completions']) < self.max_attempts_generator + self.max_attempts_verifier:
                last_step = self.call_llm(system_prompt, messages, type=type, llm_options=llm_options)

                # check steps:
                result = self.check_step(problem, solution, steps, last_step)
                rating = result['result']
                if rating != -1:
                    passed_check = True
                    step_data["chosen_index"] = index
                    step_data["model"] = self.model if type == 'generator' else self.model_verifier

                step_data["completions"].append({
                    "step": last_step,
                    "rating": rating,
                    "explanation": result['explanation']
                })

                if not passed_check and len(step_data['completions']) == self.max_attempts_generator:
                    # use verifier to complete message
                    type = 'verifier'

                index += 1

            new_message = {"role": "assistant", "content": last_step}
            messages.append(new_message)

            dataset_steps.append(step_data)

        # compare solutions
        generated_solution = parse_messages(messages)[-1]
        is_solution_correct = self.compare_solutions(problem, solution, generated_solution)

        dataset = {
            "problem": problem,
            "solution": solution,
            "steps": dataset_steps,
            "generated_solution": generated_solution,
            "is_solution_correct": is_solution_correct
        }

        return dataset

    def check_step(self, problem, solution, steps, last_step):
        steps_str = ''
        for index, step in enumerate(steps):
            steps_str += f"Step {index}:\n{step}\n\n"
        steps_str += f"Step {len(steps)} (Last Step):\n{last_step}"

        with open('prompts/prompt_find_errors_laststep.txt') as f:
            prompt = f.read()
        prompt = prompt.format(problem=problem, solution=solution, steps_str=steps_str)

        system_prompt = 'You are a mathematician and former Putnam gold medalist. You respond only in JSON.'
        llm_options = {"response_format": {"type": "json_object"}}

        messages = self.run_conversation([prompt], type='verifier', system_prompt=system_prompt, llm_options=llm_options)
        return json.loads(messages[-1]['content'])

    def compare_solutions(self, problem, solution_actual, solution_generated):
        with open('prompts/prompt_compare.txt') as f:
            prompts = f.read().split('\n\n\n')
        prompts[0] = prompts[0].format(problem=problem, solution_actual=solution_actual, solution_generated=solution_generated)

        messages = self.run_conversation(prompts, type="verifier")
        last_message = messages[-1]['content']
        if last_message.startswith('FIRST'):
            return False
        else:
            return True

    def get_conversation_prompts(self, problem, insights=None):
        with open('prompts/prompt_conversation.txt') as f:
            prompts = f.read().split('\n\n\n')
        prompts[0] = prompts[0].format(problem=problem, insights=insights)
        return prompts

    def get_insights(self, problem, solution):
        with open('prompts/prompt_insights.txt') as f:
            prompt = f.read()
            prompt = prompt.format(problem=problem, solution=solution)

        result = self.run_conversation([prompt], type="verifier")
        insights = result[-1]['content']
        return insights

    def add_to_dataset(self, steps, steps_annotated, data, generated_solution, model, model_verifier, is_solution_correct):
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

        self.dataset_steps.append({
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

    def expand_problem(self, problem, solution, llm_options={}):
        insights = self.get_insights(problem, solution)
        prompts = self.get_conversation_prompts(problem, insights=insights)
        dataset = self.run_conversation_with_checks(prompts, problem, solution, llm_options=llm_options)
        return dataset

    def expand_dataset(self, cutoff=None, llm_options={}, save=True, save_path=''):
        dataset_steps = []
        for index, data in enumerate(self.dataset):
            start_time = time.time()
            if cutoff is not None and index > cutoff:
                break

            problem = data['problem']
            solution = data['solution']

            problem_steps = self.expand_problem(problem, solution, llm_options=llm_options)

            stop_time = time.time()
            time_elapsed = stop_time - start_time

            problem_steps['year'] = data['year']
            problem_steps['label'] = data['label']
            problem_steps['id'] = f"{data['year']}_{data['label']}"
            problem_steps['model'] = self.model
            problem_steps['model_verifier'] = self.model_verifier
            problem_steps['time_elapsed'] = time_elapsed

            dataset_steps.append(problem_steps)

            print(f"Problem {index} took {time_elapsed} seconds")

            # save dataset incrementally
            if save:
                with open(save_path, 'w') as f:
                    json.dump(dataset_steps, f, indent=4)

if __name__ == "__main__":
    save = True
    options = [
        "gpt-4-1106-preview",
        "gpt-3.5-turbo",
        "mistral-7b-instruct",
        "mixtral-8x7b-instruct",
        "llama2-70b-chat",
        "code-llama-34b-instruct",
        "llama2-13b-chat",
        "llama2-70b-instruct",
        "wizardlm-70b",
        "wizardlm-13b",
        "llema-7b",
        "deepseek-chat",
        "deepseek-coder",
        "claude3"]

    model_nick = "mixtral-8x7b-instruct"
    provider = 'together'
    # model_nick = "deepseek-chat"

    max_attempts_generator = 4
    max_attempts_verifier = 2
    llm_options = {"temperature": 0.5}
    cutoff = 100

    # Options: ["openai", "perplexity", "together", "deepseek", "anthropic"]

    provider_verifier = 'openai'
    model_verifier_nick = "gpt-4-1106-preview"

    # provider_verifier = 'anthropic'
    # model_verifier_nick = "claude3-sonnet"
    # model_verifier_nick = "claude3-opus"

    # load jsonl dataset into array
    with open('datasets/math_competitions/putnam.jsonl') as f:
        dataset = [json.loads(line) for line in f]

    date = datetime.now().strftime("%Y-%m-%d")
    save_path = f'datasets_synthetic/putnam_steps_g_{model_nick}_v_{model_verifier_nick}_{date}.json'
    expander = DatasetExpander(
        dataset,
        provider,
        provider_verifier,
        model_nick,
        model_verifier_nick,
        max_attempts_generator=max_attempts_generator,
        max_attempts_verifier=max_attempts_verifier)

    expander.expand_dataset(cutoff=cutoff, llm_options=llm_options, save=save, save_path=save_path)
