import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import torch

from accelerate import Accelerator
from accelerate.utils import set_seed

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
from preprocess.utils import get_item_text
from preprocess.template import user2item_template
from src.huggingface_model_infer import run_model_embedding


def get_preference_ranks_large(scores, preferred_items, chunk_size=100000):
    M, N = scores.shape
    preferred_items = torch.tensor(preferred_items, device=scores.device)
    ranks = torch.empty(M, dtype=torch.long, device=scores.device)
    
    for start in tqdm(range(0, M, chunk_size), desc='get_preference_ranks_large'):
        end = min(start + chunk_size, M)
        chunk_scores = scores[start:end]
        chunk_preferred_items = preferred_items[start:end]
    
        sorted_indices = chunk_scores.argsort(dim=1, descending=True)
        row_indices = torch.arange(start, end, device=scores.device)
        chunk_ranks = (sorted_indices == chunk_preferred_items.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        ranks[start:end] = chunk_ranks
    
    return ranks

def gen_retrieval_result(args, item_embedding_path, user_embedding_path, itemid2text):
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    item_embeddings = torch.tensor(pickle.load(open(item_embedding_path, "rb")))
    user_embeddings = torch.tensor(pickle.load(open(user_embedding_path, "rb")))
    # user_embeddings1 = torch.tensor(pickle.load(open(xx, "rb")))
    # user_embeddings2 = torch.tensor(pickle.load(open(xx, "rb")))
    # user_embeddings = torch.cat([user_embeddings1, user_embeddings2], 0)
    print("shape of item embeddings: ", item_embeddings.shape)
    print("shape of user embeddings: ", user_embeddings.shape)

    scores = torch.softmax(torch.matmul(user_embeddings, item_embeddings.T), -1).squeeze()
    del item_embeddings
    del user_embeddings
    # del user_embeddings1
    # del user_embeddings2
    targets = []
    all_data = []
    for idx, line in tqdm(enumerate(open(args.user_embedding_prompt_path)), desc='load user embedding prompt'):
        data = json.loads(line)
        all_data.append(data)
        targets.append(data['item_id'])
        scores[idx][0] = -2
        scores[idx][data["history"]] = -2

    ranks = get_preference_ranks_large(scores, targets) 
    sorted_ranks, sorted_indices = torch.sort(ranks, descending=True)
    selected_indices = sorted_indices[:args.n_samples].tolist()
    pickle.dump(ranks, open(os.path.dirname(args.user_embedding_prompt_path)+"/ranks.pkl", "wb"))
    pickle.dump(selected_indices, open(os.path.dirname(args.user_embedding_prompt_path)+"/selected_indices.pkl", "wb"))
    # a=pickle.load(open(os.path.dirname(args.user_embedding_prompt_path)+"/ranks.pkl", "rb"))
    # b=pickle.load(open(os.path.dirname(args.user_embedding_prompt_path)+"/selected_indices.pkl", "rb"))

    with open(args.answer_file, "w", encoding='utf-8') as fd:
        for idx in tqdm(selected_indices, desc='generate answer file'):
            ground_set = set(all_data[idx]['all_history'])
            neg_items = []
            while len(neg_items) < 7:
                neg_item = random.randint(1, len(itemid2text)-1)
                if neg_item not in ground_set:
                    neg_items.append(neg_item)
            line = {
                'user_id': all_data[idx]['user_id'], 
                'item_id': all_data[idx]['item_id'],
                'neg_ids': neg_items,
                'query': all_data[idx]['text'],
                'pos': [itemid2text[all_data[idx]['item_id']]],
                'neg': [itemid2text[x] for x in neg_items],
            }
            fd.write(json.dumps(line)+'\n')
    
    

def parse_args():
    parser = argparse.ArgumentParser(description="infer selection")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--model_path_or_name", type=str, help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--exist_user_prompt_path", type=str, help="Path to query file"
    )
    parser.add_argument(
        "--user_embedding_prompt_path", type=str, help="Path to query file"
    )
    parser.add_argument(
        "--answer_file", type=str, help=""
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help=""
    )
    parser.add_argument(
        "--query_max_len", type=int, default=512, help=""
    )
    parser.add_argument(
        "--passage_max_len", type=int, default=128, help=""
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=128, help=""
    )
    parser.add_argument(
        "--sentence_pooling_method", type=str, default='cls', help="the pooling method, should be cls, mean or last", choices=['cls', 'mean', 'last']
    )
    parser.add_argument(
        "--normlized", action='store_true', help=""
    )
    parser.add_argument(
        "--has_template", action='store_true', help=""
    )
    parser.add_argument(
        "--peft_model_name", type=str, default=None, help=""
    )
    parser.add_argument(
        "--torch_dtype", type=str, default=None, help="", choices=["auto", "bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--n_samples", type=int, default=150000, help="select n hard samples"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()

    ## get the dir of args.user_embedding_prompt_path
    cache_dir =  os.path.dirname(args.user_embedding_prompt_path)
    item_embedding_prompt_path = os.path.join(cache_dir, 'item_embedding_prompt.jsonl')
    item_embedding_path = os.path.join(cache_dir, 'item_embedding.pkl')
    user_embedding_path = os.path.join(cache_dir, 'user_embedding.pkl')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, use_fast=True)

    if not os.path.exists(item_embedding_path):
        if accelerator.is_main_process:
            os.makedirs(cache_dir, exist_ok=True)
            get_item_text(args.in_meta_data, save_item_prompt_path=item_embedding_prompt_path)
        accelerator.wait_for_everyone()
        
        accelerator.print("infer item embedding")
        run_model_embedding(args.model_path_or_name, max_seq_len=args.passage_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=item_embedding_prompt_path, emb_path=item_embedding_path, accelerator=accelerator, args=args, qorp='passage')


    itemid2text, itemid2title, itemid2features, itemid2price_date_map = get_item_text(args.in_meta_data)

    if not os.path.exists(args.user_embedding_prompt_path):
        if accelerator.is_main_process:
            exist_user_item_pairs = set()
            with open(args.exist_user_prompt_path, 'r') as rd:
                for line in rd:
                    data = json.loads(line)
                    exist_user_item_pairs.add((int(data['user_id']), int(data['item_id'])))
            
            candidate_query = []
            with open(args.in_seq_data, 'r') as rd:
                for idx, line in tqdm(enumerate(rd), desc='generate user_embedding_prompt_path'):
                    userid, itemids = line.strip().split(' ', 1)
                    itemids = itemids.split(' ')
                    for i, itemid in enumerate(itemids[2:-1]):
                        if (int(userid), int(itemid)) not in exist_user_item_pairs:
                            cand = {'user_id': int(userid), 'item_id': int(itemid), 'history': [int(x) for x in itemids[:i+2]], 'all_history': [int(x) for x in itemids]}
                            query_items = itemids[:i+2][::-1]
                            query_items = query_items[:20]
                            if random.random() < 0.5:
                                template = "{}"
                            else:
                                template = random.choice(user2item_template)
                            query = ''
                            for x in query_items:
                                query += itemid2title[int(x)][1] + ', '
                            query = query.strip().strip(',')
                            template_length = len(tokenizer.tokenize(template))
                            tokens = tokenizer.tokenize(query)[:args.query_max_len-template_length]
                            truncated_query = tokenizer.convert_tokens_to_string(tokens).strip().strip(',')

                            query = template.format(truncated_query)
                            cand['text'] = query
                            candidate_query.append(cand)
            with open(args.user_embedding_prompt_path, 'w') as fd:
                for line in candidate_query:
                    fd.write(json.dumps(line)+'\n')

        accelerator.wait_for_everyone()

    if not os.path.exists(user_embedding_path):
        accelerator.print("infer user embedding")
        run_model_embedding(args.model_path_or_name, max_seq_len=args.query_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=args.user_embedding_prompt_path, emb_path=user_embedding_path, accelerator=accelerator, args=args, qorp='query')

    if accelerator.is_main_process:
        gen_retrieval_result(args, item_embedding_path, user_embedding_path, itemid2text)