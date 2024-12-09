import json
import random
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
import time
from template import rewrite_template
from utils import get_item_text, get_model

random.seed(2024)

def parse_args():
    parser = argparse.ArgumentParser(description="rewrite item")
    parser.add_argument(
        "--in_meta_data", type=str, default='', help=""
    )
    parser.add_argument(
        "--out_meta_data", type=str, default='', help=""
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default='', help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help=""
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help=""
    )
    parser.add_argument(
        "--end_idx", type=int, default=3000, help=""
    )
    args = parser.parse_args()
    return args

def run(args, client, tokenizer):
    input_token_num, output_token_num = 0, 0
    itemid2text, itemid2title, itemid2features, itemid2price_date_map = get_item_text(args.in_meta_data)
    raw_text = []
    rewrite_text = []
    for i, item_text in enumerate(tqdm(itemid2text[args.start_idx:args.end_idx])):
        if args.start_idx==0 and i==0:
            raw_text.append(item_text)
            rewrite_text.append(item_text)
            continue
        while True:
            try:
                if args.model_name_or_path.startswith("gpt"):
                    time.sleep(0.3)
                response = client.chat.completions.create(
                    model=args.model_name_or_path,
                    messages=[
                        {"role": "system",
                        "content": "You are a helpful assistant. \n"},
                        {"role": "user", "content": rewrite_template.format(target_info=item_text)},
                    ],
                    max_tokens=600,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                tokens = response.usage
                response_text = response.choices[0].message.content
                input_token_num += int(tokens.prompt_tokens)
                output_token_num += int(tokens.completion_tokens)
                if response_text.lower().startswith('title'):
                    continue
                rewrite_text.append(response_text)
                raw_text.append(item_text)
                break
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                print(f'i: {i}')
                return rewrite_text, raw_text, input_token_num, output_token_num
            except Exception as e:
                print(f'error: {e}')
                print(f'i: {i}')
                return rewrite_text, raw_text, input_token_num, output_token_num
    return rewrite_text, raw_text, input_token_num, output_token_num


if __name__ == '__main__':
    args = parse_args()
    client, tokenizer = get_model(args)
    rewrite_text, raw_text, input_token_num, output_token_num = run(args, client, tokenizer)
    
    with open(args.out_meta_data, 'a') as f:
        for raw, rewrite in zip(raw_text, rewrite_text):
            line = {'raw_text': raw, 'rewrite_text': rewrite}
            f.write(json.dumps(line)+'\n')

    cost = 5 * input_token_num / 1000000 + 15 * output_token_num / 1000000
    print(f'cost: {cost} USD')
    print(f"input token num: {input_token_num}, output token num: {output_token_num}")
