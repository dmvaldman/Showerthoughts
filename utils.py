import os
from openai import OpenAI

def get_model_path(model, provider):
    if provider == 'openai':
        return model

    if model == 'mistral-7b-instruct':
        if provider == 'perplexity':
            model_name = 'mistral-7b-instruct'
        elif provider == 'together':
            model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    elif model == 'mixtral-8x7b-instruct':
        if provider == 'together':
            model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        elif provider == 'perplexity':
            model_name = 'mixtral-8x7b-instruct'
    elif model == 'llama2-70b-chat':
        if provider == 'perplexity':
            model_name = 'llama-2-70b-chat'
        elif provider == 'together':
            model_name = 'togethercomputer/llama-2-70b-chat'
    elif model == 'code-llama-34b-instruct':
        if provider == 'together':
            model_name = 'togethercomputer/CodeLlama-34b-Instruct'
        elif provider == 'perplexity':
            model_name = 'codellama-34b-instruct'
    elif model == 'llama2-13b-chat':
        if provider == 'perplexity':
            raise NotImplementedError
        elif provider == 'together':
            model_name = 'togethercomputer/llama-2-13b-chat'
    elif model == 'llama2-70b-instruct':
        if provider == 'perplexity':
            raise NotImplementedError
        elif provider == 'together':
            model_name = 'llama2-70b'
    elif model == 'wizardlm-70b':
        if provider == 'perplexity':
            raise NotImplementedError
        elif provider == 'together':
            model_name = 'WizardLM/WizardLM-70B-V1.0'
    elif model == 'wizardlm-13b':
        if provider == 'perplexity':
            raise NotImplementedError
        elif provider == 'together':
            model_name = 'WizardLM/WizardLM-13B-V1.2'
    elif model == 'llema-7b':
        if provider == 'perplexity':
            raise NotImplementedError
        elif provider == 'together':
            model_name = 'EleutherAI/llemma_7b'
    else:
        raise Exception(f"Model {model} not supported")

    return model_name


def get_client(provider):
    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = "https://api.openai.com/v1"
    elif provider == "together":
        api_key = os.getenv('TOGETHER_API_KEY')
        base_url = "https://api.together.xyz/"
    elif provider == "perplexity":
        api_key = os.getenv('PERPLEXITY_API_KEY')
        base_url = "https://api.perplexity.ai"

    client = OpenAI(
        api_key = api_key,
        base_url = base_url
    )

    return client