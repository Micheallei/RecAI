import argparse
import json
import csv
import pandas as pd
import os
import time
import random
import numpy as np
import torch
from openai import OpenAI, AzureOpenAI
from azure.identity import get_bearer_token_provider, AzureCliCredential
import tiktoken
from tqdm import tqdm
from transformers import AutoTokenizer

seed = 2024
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Simulator")
    parser.add_argument(
        "--model_name_or_path", type=str, default='', help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--query_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--response_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--max_output_token_length", type=int, default=512, help=""
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help=""
    )
    parser.add_argument(
        "--end_idx", type=int, default=100000000, help=""
    )
    args = parser.parse_args()
    return args

def get_model(args):
    api_key = os.environ.get('OPENAI_API_KEY') if os.environ.get('OPENAI_API_KEY') else None
    api_base =  os.environ.get('OPENAI_API_BASE') if os.environ.get('OPENAI_API_BASE') else None
    api_type = os.environ.get('OPENAI_API_TYPE') if os.environ.get('OPENAI_API_TYPE') else None
    api_version =  os.environ.get('OPENAI_API_VERSION') if os.environ.get('OPENAI_API_VERSION') else None

    if api_key:
        if api_type == "azure":
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base,
            )
        else:
            client = OpenAI(  
                api_key=api_key,
                base_url=api_base,
            )
    else:
        credential = AzureCliCredential()    

        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=api_base,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            max_retries=5,
        )

    if args.model_name_or_path.startswith("gpt-3"):
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif args.model_name_or_path.startswith("gpt-4o"):
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    elif args.model_name_or_path.startswith("gpt-4"):
        tokenizer = tiktoken.encoding_for_model("gpt-4")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    return client, tokenizer

def load_jsonl_from_disk(file_path):
    ## read a csv file with pd, the first line is header 
    df = pd.read_csv(file_path)    
    return df.to_dict(orient='records')

class API():
    def __init__(self, args):
        self.args = args
        self.client, self.tokenizer = get_model(args)
        self.query_data = load_jsonl_from_disk(args.query_file)
        self.input_token_num = 0
        self.output_token_num = 0
        if os.path.exists(args.response_file):
            self.flag=1
        else:
            self.flag=0

    def run(self):
        results = []
        for i, sample in tqdm(enumerate(self.query_data[self.args.start_idx:self.args.end_idx]), desc='Running', total=len(self.query_data[self.args.start_idx:self.args.end_idx])):
            response = self.run_once(sample, i)
            if response is None:
                print(f"i: {i}, sample: {sample}")
                break
            results.append([sample['question'], response])

        with open(self.args.response_file, 'a') as csv_file:
            writer = csv.writer(csv_file)
            if self.flag==0:
                writer.writerow(['question', 'response'])
            for result in results:
                writer.writerow([result[0], result[1]])
        
        cost = 5 * self.input_token_num / 1000000 + 15 * self.output_token_num / 1000000
        print(f'cost: {cost} USD')
        print(f"input token num: {self.input_token_num}, output token num: {self.output_token_num}")

    def run_once(self, sample, i):
        try:
            if self.args.model_name_or_path.startswith("gpt"):
                time.sleep(0.5)
            response = self.client.chat.completions.create(
                model=self.args.model_name_or_path,  
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. \n"},
                    {"role": "user", "content": sample['question']},
                ],
                max_tokens=self.args.max_output_token_length,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
            )
            tokens = response.usage
            response = response.choices[0].message.content
            self.input_token_num += int(tokens.prompt_tokens)
            self.output_token_num += int(tokens.completion_tokens)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            return None
        except Exception as e:
            print(f'error: {e}')
            # print(f'prompt: {prompt}')
            return None
        return response


if __name__ == '__main__':
    args = parse_args()
    llm_api = API(args)
    llm_api.run()