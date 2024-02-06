# Combines and filters all the smaller datasets in the datasets directory into a single final one

import openai
import os
import json
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import re
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import tkinter as tk
from PIL import Image, ImageTk


client = openai.Client()

def dedupe(df, threshold=0.85, plot=False, save=False):
    corpus_problems = df['problem'].tolist()
    embeddings_problem = embed(corpus_problems, save=save, save_path="datasets/combined_embeddings.npy")

    index_neighbors = create_faiss_index(embeddings_problem)
    D_problems, I_problems = index_neighbors.search(embeddings_problem, k=4)

    if plot:
        nearest_neighbor_distances = D_problems[:, 1]
        plt.figure(figsize=(10, 6))
        plt.hist(nearest_neighbor_distances, bins=50, alpha=0.75)
        plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.title('Distribution of Nearest Neighbor Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    dupe_indices = []
    for index, distances in enumerate(D_problems):
        # distances[0] is the distance to the point itself, distances[1] is the nearest neighbor
        for index_neighbors, is_neighbor_distance_below_thresh in enumerate(distances > threshold):
            # skip first distance as it is itself
            if index_neighbors == 0:
                continue

            dupe_index = I_problems[index, index_neighbors]
            if is_neighbor_distance_below_thresh and index < dupe_index:
                is_from_diff_datasets = df['source'][index] != df['source'][dupe_index]
                if not is_from_diff_datasets:
                    continue
                distance = distances[index_neighbors]
                dupe_indices.append((index, dupe_index, distance))
                # print(f"Dupe: (p:{distance})\n{index}: {corpus_problems[index]}\n{dupe_index}: {corpus_problems[dupe_index]}\n\n")

    return dupe_indices

def filter(dataset):
    # if problem has https://cdn.mathpix.com/ in it, remove it
    dataset = dataset[~dataset['problem'].str.contains('https://cdn.mathpix.com/')]
    # this problem contains an images, references an image in the solution, is it solvable without access to the image?


def filter_img_urls(problems, dir='images', save=False, save_path=''):
    pattern = r'https://cdn\.mathpix\.com[^\s)]*'
    urls = []
    for index, problem in enumerate(problems):
        found_urls = re.findall(pattern, problem)
        if found_urls:
            for url in found_urls:
                urls.append((index, url))

    responses = {}
    for index, url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            # print problem
            print(f'#{index}: {problems[index]})')

            # caption is the problem with the url removed
            caption = problems[index].replace('\n![](' + url + ')', '')

            # Display the image from bytes
            key = display_image_and_get_response(response.content, caption)

            if key == 'k':
                responses[index] = True
            elif key == 'd':
                responses[index] = False
            else:
                print(f"Invalid key: {key}")

            # save image to dir
            filepath = os.path.join(dir, str(index) + '.jpg')
            with open(filepath, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download image from {url}")

    if save:
        with open(save_path, 'w') as f:
            json.dump(responses, f)

    return responses

def display_image_and_get_response(image_bytes, caption):
    """
    Displays an image and a caption, waits for 'k' or 'd' key press.

    :param image_bytes: Byte array of the image to display.
    :param caption: Caption to display with the image.
    :return: 'k' if confirmed, 'd' if denied.
    """
    response = {'value': None}
    caption += '\n\nPress "k" to keep, "d" to delete.'

    def on_key_press(event):
        """Handle key press event."""
        nonlocal response
        if event.char == 'k':
            response['value'] = 'k'
            root.destroy()
        elif event.char == 'd':
            response['value'] = 'd'
            root.destroy()

    # Set up the Tkinter window
    root = tk.Tk()
    root.title("Image Confirmation")

    # Convert image bytes to Tkinter-compatible format
    image = Image.open(BytesIO(image_bytes))
    photo = ImageTk.PhotoImage(image)

    # Add image to the Tkinter window
    label_image = tk.Label(root, image=photo)
    label_image.pack()

    # Add caption below the image
    label_caption = tk.Label(root, text=caption, wraplength=400)
    label_caption.pack()

    # Bind key press event
    root.bind("<Key>", on_key_press)

    # Start the GUI event loop
    root.mainloop()

    return response['value']

def embed(corpus, chunk_size=1000, save=False, save_path=''):
    # first load
    if os.path.exists(save_path):
        print(f"loading embeddings from disk at {save_path}")
        return np.load(save_path)

    embeddings = []
    for i in range(0, len(corpus), chunk_size):
        chunk = corpus[i:i + chunk_size]
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        for embed in response.data:
            embeddings.append(embed.embedding)

    embeddings = np.array(embeddings)

    if save:
        np.save(save_path, embeddings)

    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Dimension of the embeddings
    # Using cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)  # Adding the embeddings to the index
    return index

def gather(dataset_dir, save_path=None):
    # combine all files ending in jsonl extension that are within a directory
    files = []
    for root, dirs, file in os.walk(dataset_dir):
        for f in file:
            if f.endswith('.jsonl'):
                filepath = os.path.join(root, f)
                # basename without extension
                name = os.path.basename(filepath).split('.')[0].capitalize()
                dataset_dir = root.split('/')[-1]
                dataset_dir = dataset_dir.replace('_', ' ')
                files.append((dataset_dir, name, filepath))

    # read all files and combine them into a single file
    # {problem, solution, hint, answer, source, type}
    open(save_path, 'w').close()
    with open(save_path, 'a') as outfile:
        for dataset_dir, name, fname in files:
            if fname == save_path:
                continue
            with open(fname) as infile:
                for line in infile:
                    data = json.loads(line)

                    if 'instruction' in data:
                        new_data = {
                            'type': dataset_dir,
                            'source': name,
                            'problem': data['instruction'] + data['input'],
                            'solution': data['output'],
                            'hint': '',
                            'answer': data.get('answer', '')
                        }
                    else:
                        new_data = {
                            'type': dataset_dir,
                            'source': name,
                            'problem': data['problem'],
                            'solution': data['solution'],
                            'hint': data.get('hint', ''),
                            'answer': data.get('answer', '')
                        }

                    # if no solution, use hint or answer
                    if new_data['solution'] is None:
                        if new_data['hint'] is not None:
                            new_data['solution'] = 'Hint: ' + new_data['hint']
                        else:
                            new_data['solution'] = new_data['answer']

                    # write to jsonl
                    outfile.write(json.dumps(new_data) + '\n')

save_path = 'datasets/combined.jsonl'
# combined all datasets in the dataset directory
# gather("datasets", save_path=save_path)

# Load the dataset
df = pd.read_json(save_path, lines=True)

# Filter dataset for images
# This will run a simple GUI asking to confirm/deny if the image is necessary to solve the problem.
# If not, we can remove the reference, if so, we can delete the row to keep things text-only
filter_indices_problems = filter_img_urls(df['problem'], dir='images/images_problems', save=True, save_path = 'datasets/filter_indices_problems.json')
filter_indices_solutions = filter_img_urls(df['solution'], dir='images/images_solutions', save=True, save_path = 'datasets/filter_indices_solutions.json')

# Dedupe repeating problems using cosine similarity
dupe_indices_data = dedupe(df, threshold=0.85, plot=False, save=True)
dupe_indices = [dupe_index for _, dupe_index, _ in dupe_indices_data]
df_deduped = df.drop(df.index[dupe_indices])
