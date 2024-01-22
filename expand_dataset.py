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
    def __init__(self, dataset, provider, model, model_verifier):
        self.dataset = dataset
        self.provider = provider

        # generator model
        self.model = get_model_path(model, provider)
        self.client = get_client(provider)

        # verifier model
        self.model_verifier = get_model_path(model_verifier, "openai")
        self.client_verifier = get_client("openai")

        self.dataset_steps = []

    @retry(stop=stop_after_attempt(3))
    def call_llm(self, messages, type, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, response_format=None):
        if type == 'generator':
            method = self.call_llm_generator
        elif type == 'verifier':
            method = self.call_llm_verifier

        finished = False
        reply = ''
        while not finished:
            response = method(messages, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, response_format=response_format)
            finish_reason = response.choices[0].finish_reason

            if finish_reason == 'length':
                raise Exception("Conversation too long")
            elif finish_reason == "content_filter":
                raise Exception("Conversation filtered")

            # finish reason not implemented in some providers
            finished = (finish_reason == "stop") or (finish_reason is None)
            reply += response.choices[0].message.content

        return reply

    def call_llm_generator(self, messages, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, response_format=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format
        )
        return response

    @retry(stop=stop_after_attempt(3))
    def call_llm_verifier(self, messages, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, response_format=None):
        response = self.client_verifier.chat.completions.create(
            model=self.model_verifier,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format
        )
        return response

    def run_conversation(self, prompts, system_prompt=None, type='generator', temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, response_format=None):
        if system_prompt is None:
            if type=='generator':
                system_prompt = DatasetExpander.system_prompt
            elif type=='verifier':
                system_prompt = DatasetExpander.system_prompt_verifier
            else:
                raise Exception("System prompt not specified")

        messages=[
            {"role": "system", "content": system_prompt}
        ]

        for prompt in prompts:
            message = {"role": "user", "content": prompt}
            messages.append(message)

            reply = self.call_llm(messages, type=type, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, response_format=response_format)
            new_message = {"role": "assistant", "content": reply}

            messages.append(new_message)

        return messages

    def compare_solutions(self, problem, solution1, solution2):
        with open('prompts/prompt_compare.txt') as f:
            prompts = f.read().split('\n\n\n')
        prompts[0] = prompts[0].format(problem=problem, solution1=solution1, solution2=solution2)

        messages = self.run_conversation(prompts)
        last_message = messages[-1]['content']
        return last_message

    def get_conversation_prompts(self, problem, insights=None):
        with open('prompts/prompt_conversation.txt') as f:
            prompts = f.read().split('\n\n\n')
        prompts[0] = prompts[0].format(problem=problem, insights=insights)
        return prompts

    def find_steps_with_error(self, problem, solution, steps, model=None):
        steps_str = ''
        for index, step in enumerate(steps):
            steps_str += f"Step {index}:\n{step}\n\n"
        steps_str = steps_str.strip()

        with open('prompts/prompt_find_errors.txt') as f:
            prompt = f.read()
        prompt = prompt.format(problem=problem, solution=solution, steps_str=steps_str)

        system_prompt = 'You are a mathematician and former Putnam gold medalist. You respond only in JSON.'

        response = self.run_conversation([prompt], type='verifier', system_prompt=system_prompt, response_format={"type": "json_object"})
        return json.loads(response[-1]['content'])


    def get_insights(self, problem, solution):
        with open('prompts/prompt_insights.txt') as f:
            prompt = f.read()
            prompt = prompt.format(problem=problem, solution=solution)

        result = self.run_conversation([prompt], type="verifier")
        insights = result[-1]['content']
        return insights

    def get_steps_and_solution(self, problem, insights=None):
        prompts = self.get_conversation_prompts(problem, insights=insights)
        messages = self.run_conversation(prompts, type='generator')
        steps = parse_messages(messages)
        generated_solution = steps[-1]
        return steps[0:-1], generated_solution

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

    def expand_problem(self, problem, solution):
        insights = self.get_insights(problem, solution)
        steps, generated_solution = self.get_steps_and_solution(problem, insights=insights)
        choice = self.compare_solutions(problem, solution, generated_solution)
        is_solution_correct = (choice == 'SECOND' or choice == 'BOTH')
        steps_annotated = self.find_steps_with_error(problem, solution, steps)

        return steps, steps_annotated, generated_solution, is_solution_correct

    def expand_dataset(self, save=True, save_path=''):
        for index, data in enumerate(self.dataset):
            if index > 10:
                break

            problem = data['problem']
            solution = data['solution']

            steps, steps_annotated, generated_solution, is_solution_correct = self.expand_problem(problem, solution)
            self.add_to_dataset(steps, steps_annotated['steps'], data, generated_solution, self.model, self.model_verifier, is_solution_correct)

            # save dataset
            if save:
                with open(save_path, 'w') as f:
                    json.dump(self.dataset_steps, f, indent=4)

if __name__ == "__main__":
    # Options: ["gpt-4-1106-preview", "gpt-3.5-turbo", "mistral-7b-instruct", "mixtral-8x7b-instruct", "llama2-70b-chat", "code-llama-34b-instruct", "llama2-13b-chat", "llama2-70b-instruct", "wizardlm-70b", "wizardlm-13b", "llema-7b"]
    model_nickname = "mixtral-8x7b-instruct"
    model_verifier_nickname = "gpt-4-1106-preview"

    # Options: ["openai", "perplexity", "together"]
    provider = 'together'

    # load dataset
    with open('datasets/putnam/putnam.json') as f:
        dataset = json.load(f)

    save_path = f'datasets/putnam/putnam_steps_{model_nickname}.json'
    expander = DatasetExpander(dataset, provider, model_nickname, model_verifier_nickname)
    expander.expand_dataset(save=True, save_path=save_path)
