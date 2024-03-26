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
from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageFont
import hashlib
from weasyprint import HTML
import base64


client = openai.Client()

def dedupe(df, threshold=0.85, plot=False, save=False, force=False):
    corpus_problems = df['problem'].tolist()
    embeddings_problem = embed(corpus_problems, save=save, save_path="datasets/combined_embeddings.npy", force=force)

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

    dupe_indices = [dupe_index for _, dupe_index, _ in dupe_indices]
    df_deduped = df.drop(df.index[dupe_indices])

    return df_deduped

def save_window(root, filename="saved_window.png"):
    root.update_idletasks()  # Update "idle" tasks to ensure geometry is updated
    root.withdraw()  # Hide the window as we don't want it to appear in the screenshot
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    x1 = x + root.winfo_width()
    y1 = y + root.winfo_height()
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    root.deiconify()  # Show the window again

def filter(dataset):
    # if problem has https://cdn.mathpix.com/ in it, remove it
    dataset = dataset[~dataset['problem'].str.contains('https://cdn.mathpix.com/')]
    # this problem contains an images, references an image in the solution, is it solvable without access to the image?

def create_captioned_image(image_bytes, caption, save=False, save_path=''):
    # we append _keep, _remove after the fact to the path
    save_path_firstpart = save_path.split('_')[0]

    if os.path.exists(save_path_firstpart):
        print(f"Image already exists at {save_path}")
        return

    encoded_image_data = base64.b64encode(image_bytes).decode('utf-8')

    # sanitize caption for html
    caption = caption.replace('<', '&lt;')
    caption = caption.replace('>', '&gt;')

    # Create an HTML string with your image and caption
    html_template = f"""
    <html>
    <head>
        <style>
            body {{ text-align: center; }}
            img {{ max-width: 100%; }}
            .caption {{ font-family: Arial, sans-serif; margin-top: 10px; font-size: 1.3em;}}
        </style>
    </head>
    <body>
        <img src="data:image/png;base64,{encoded_image_data}" alt="captioned image">
        <div class="caption">{caption}</div>
    </body>
    </html>
    """

    # Convert HTML to pdf
    if save:
        HTML(string=html_template).write_pdf(save_path)

def extract_images(df, dir='images', save=False):
    column_names = ['problem', 'solution']
    pattern = r'https://cdn\.mathpix\.com[^\s)]*'

    for column_name in column_names:
        urls = []
        for row in df.itertuples():
            # get column name
            text = getattr(row, column_name)
            id = row.id
            # find first occurence of a url
            found_urls = re.findall(pattern, text)
            if found_urls:
                urls.append((id, found_urls[0], text))

        for id, url, text in urls:
            response = requests.get(url)
            if response.status_code == 200:
                image_bytes = response.content
            else:
                print(f"Failed to download image from {url}")
                continue

            # caption is the problem with the url removed
            caption = text.replace('\n![](' + url + ')', '')

            # create image + caption
            save_path = os.path.join(dir, column_name, str(id) + '.pdf')
            create_captioned_image(image_bytes, caption, save=save, save_path=save_path)

def embed(corpus, chunk_size=1000, save=False, save_path='', force=False):
    # first load
    if os.path.exists(save_path) and not force:
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
            if fname == save_path or fname.startswith('datasets/combined'):
                continue
            with open(fname) as infile:
                for line in infile:
                    data = json.loads(line)

                    if 'instruction' in data:
                        problem = data['instruction'] + data['input']
                        solution = data['output']
                        hint = ''
                        answer = data.get('answer', '')
                    else:
                        problem = data['problem']
                        solution = data['solution']
                        hint = data.get('hint', '')
                        answer = data.get('answer', '')

                    # id is hash of the problem text
                    id = hashlib.md5(problem.encode()).hexdigest()

                    # if no solution, use hint or answer
                    if not solution:
                        if hint:
                            solution = 'Hint: ' + hint
                            if answer:
                                solution += '\nAnswer: ' + answer
                        elif answer:
                            solution = 'Answer: ' + answer
                        else:
                            continue

                    normalized_data = {
                        'id': id,
                        'type': dataset_dir,
                        'source': name,
                        'problem': problem,
                        'solution': solution,
                        'hint': hint,
                        'answer': answer
                    }

                    # write to jsonl
                    outfile.write(json.dumps(normalized_data) + '\n')

        df = pd.read_json(save_path, lines=True)
        return df

def filter_bad_images(df, dir='images'):
    # loop through image directories and extract first/second part of filename
    column_names = ['problem', 'solution']
    for column_name in column_names:
        image_dir = os.path.join(dir, column_name)
        for _, _, files in os.walk(image_dir):
            for file in files:
                file_parts = file.split('_')
                if len(file_parts) < 2:
                    continue
                id = file_parts[0]
                status = file_parts[1]

                if status == 'k':
                    # remove URL content from the dataframe row with the id
                    pattern = r'https://cdn\.mathpix\.com[^\s)]*'
                    row = df[df['id'] == id]
                    text = row[column_name].values[0]
                    text = re.sub(pattern, '', text)
                    df.loc[df['id'] == id, column_name] = text
                elif status == 'd':
                    # delete the row with the id
                    df = df[df['id'] != id]
    return df

def postprocess(df):
    # Remove the text FIGURE ##
    df['problem'] = df['problem'].str.replace(r'FIGURE \d+', '')
    df['solution'] = df['solution'].str.replace(r'FIGURE \d+', '')


def main():
    save_path = 'datasets/combined.jsonl'

    # combined all datasets in the dataset directory
    df = gather("datasets", save_path=save_path)

    # Dedupe repeating problems using cosine similarity
    df = dedupe(df, threshold=0.85, plot=False, save=True, force=False)

    # Filter dataset for images
    extract_images(df, dir='images', save=False)

    # Filter out problems with bad images
    df = filter_bad_images(df, dir='images')

    # Postprocess the dataset
    df = postprocess(df)

    # Save the final dataset
    df.to_json('datasets/combined_final.jsonl', orient='records', lines=True)

if __name__ == '__main__':
    main()