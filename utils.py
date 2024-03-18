import os
from openai import OpenAI
import anthropic

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
    elif model == 'deepseek-chat':
        if provider == 'deepseek':
            model_name = 'deepseek-chat'
        else:
            raise Exception(f"Model {model} only supported by {provider} provider")
    elif model == 'deepseek-code':
        if provider == 'deepseek':
            model_name = 'deepseek-code'
        else:
            raise Exception(f"Model {model} only supported by {provider} provider")
    elif model == 'claude3':
        if provider == 'anthropic':
            model_name = 'claude-3-sonnet-20240229'
        else:
            raise Exception(f"Model {model} only supported by {provider} provider")
    else:
        raise Exception(f"Model {model} not supported")

    return model_name


def get_client(provider):
    if provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        client = anthropic.Anthropic(
            api_key=api_key
        )
        return client

    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = "https://api.openai.com/v1"
    elif provider == "together":
        api_key = os.getenv('TOGETHER_API_KEY')
        base_url = "https://api.together.xyz/"
    elif provider == "perplexity":
        api_key = os.getenv('PERPLEXITY_API_KEY')
        base_url = "https://api.perplexity.ai"
    elif provider == "deepseek":
        api_key = os.getenv('DEEPSEEK_API_KEY')
        base_url = "https://api.deepseek.ai/v1"

    client = OpenAI(
        api_key = api_key,
        base_url = base_url
    )

    return client