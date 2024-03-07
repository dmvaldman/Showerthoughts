# Generate synthetic non-linear reasoning steps inbetween problem/solution of dataset

import json
import dotenv
from utils import get_model_path, get_client
from tenacity import retry, stop_after_attempt

dotenv.load_dotenv()

def parse_messages(messages):
    parsed_messages = []
    for message in messages:
        if message['role'] == 'assistant':
            parsed_message = message['content']
            parsed_messages.append(parsed_message)
    return parsed_messages

class DatasetExpander():
    system_prompt = "You are an insightful and intuitive mathematician. You're great at solving math problems via brainstorming and non-linear reasoning. Sometimes you go down dead-ends but eventually you uncover and synthesize the insights needed to form the correct answer."
    system_prompt_verifier = "You are a mathematician and former Putnam gold medalist. You respond only in JSON."
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

    def call_llm(self, messages, type, llm_options={}):
        if type == 'generator':
            model = self.model
            client = self.client
        elif type == 'verifier':
            model = self.model_verifier
            client = self.client_verifier

        finished = False
        reply = ''
        while not finished:
            response = self.call_llm_base(messages, model, client, llm_options)
            finish_reason = response.choices[0].finish_reason

            if finish_reason == 'length':
                raise Exception("Conversation too long")
            elif finish_reason == "content_filter":
                raise Exception("Conversation filtered")

            # finish reason not implemented in some providers
            finished = (finish_reason == "stop") or (finish_reason is None)
            reply += response.choices[0].message.content

        return reply

    @retry(stop=stop_after_attempt(3))
    def call_llm_base(self, messages, model, client, llm_options={}):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **llm_options
            )
        except Exception as e:
            print("Couldn't call LLM, retrying...", e)
            raise e
        return response

    def run_conversation(self, prompts, system_prompt=None, type='generator', llm_options={}):
        if system_prompt is None:
            if type=='generator':
                system_prompt = DatasetExpander.system_prompt
            elif type=='verifier':
                system_prompt = DatasetExpander.system_prompt_verifier

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        for prompt in prompts:
            message = {"role": "user", "content": prompt}
            messages.append(message)
            last_step = self.call_llm(messages, type=type, llm_options=llm_options)
            new_message = {"role": "assistant", "content": last_step}
            messages.append(new_message)

        return messages

    def run_conversation_with_checks(self, prompts, problem=None, solution=None, llm_options={}):
        messages = [
            {"role": "system", "content": DatasetExpander.system_prompt}
        ]

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
                last_step = self.call_llm(messages, type=type, llm_options=llm_options)

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
        llm_options = {
            "response_format": {"type": "json_object"}
        }

        response = self.run_conversation([prompt], type='verifier', system_prompt=system_prompt, llm_options=llm_options)
        return json.loads(response[-1]['content'])

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

    def expand_dataset(self, llm_options={}, save=True, save_path=''):
        dataset_steps = []
        for index, data in enumerate(self.dataset):
            if index > 10:
                break

            problem = data['problem']
            solution = data['solution']

            problem_steps = self.expand_problem(problem, solution, llm_options=llm_options)

            problem_steps['year'] = data['year']
            problem_steps['label'] = data['label']
            problem_steps['id'] = f"{data['year']}_{data['label']}"
            problem_steps['model'] = self.model
            problem_steps['model_verifier'] = self.model_verifier

            dataset_steps.append(problem_steps)

            # save dataset incrementally
            if save:
                with open(save_path, 'w') as f:
                    json.dump(dataset_steps, f, indent=4)

if __name__ == "__main__":
    Options = [
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
        "deepseek-coder"]

    # model_nick = "mixtral-8x7b-instruct"
    model_nick = "deepseek-chat"
    model_verifier_nick = "gpt-4-1106-preview"

    max_attempts_generator = 4
    max_attempts_verifier = 2
    llm_options = {"temperature": 0.5}

    # Options: ["openai", "perplexity", "together", "deepseek"]
    provider = 'deepseek'
    provider_verifier = 'openai'

    # load jsonl dataset into array
    with open('datasets/math_competitions/putnam.jsonl') as f:
        dataset = [json.loads(line) for line in f]

    save_path = f'datasets_synthetic/putnam_steps_{model_nick}.json'
    expander = DatasetExpander(
        dataset,
        provider,
        provider_verifier,
        model_nick,
        model_verifier_nick,
        max_attempts_generator=max_attempts_generator,
        max_attempts_verifier=max_attempts_verifier)

    expander.expand_dataset(llm_options=llm_options, save=True, save_path=save_path)
