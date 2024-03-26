import json
import pandas

dataset_path = "datasets_synthetic/combined_steps_g_mixtral-8x7b-instruct_v_gpt-4-1106-preview.jsonl"

# load the dataset
dataset = []
with open(dataset_path, 'r') as f:
    for line in f:
        dataset.append(json.loads(line))

preference_dataset = []

for row in dataset:
    steps = row['steps']
    problem = row['problem']
    solution = row['solution']

    prompt = problem + "\n"
    before = ''
    after = ''

    for step in steps:
        completions = step['completions']
        chosen_index = step['chosen_index']

        if chosen_index is None:
            # no solution found
            continue

        for completion in completions:
            rating = completion['rating']
            text = completion['step']

            if rating == -1:
                before = text
                chosen = completions[chosen_index]['step']
                preference_dataset.append({
                    'prompt': prompt,
                    'rejected': before,
                    'chosen': chosen
                })
            else:
                chosen = text

            prompt += chosen + "\n"

# save the preference dataset
path = 'datasets_synthetic/combined_steps_g_mixtral-8x7b-instruct_v_gpt-4-1106-preview_preference.jsonl'
open(path, 'w').close()
with open(path, 'a') as f:
    for entry in preference_dataset:
        f.write(json.dumps(entry) + '\n')


