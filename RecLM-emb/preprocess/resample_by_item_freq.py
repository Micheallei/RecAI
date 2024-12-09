import json
from collections import defaultdict
from tqdm import tqdm
import random
import argparse
from copy import deepcopy
from utils import get_item_text

random.seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description="genera_query_file")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--candidate_files", type=str, help="split by comma"
    )
    parser.add_argument(
        "--output_file", type=str, help=""
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    candidate_files_query = [ #The data field only has query, pos, neg
        'gpt4_data', 
        'gpt4_data_v2',
        'gpt4o_data_v2',
        'gpt4o_data',
        'llama3.1_70B_data_v2',
        'llama3.1_70B_data',
        'Qwen2.5_72B_data_v2',
        'Qwen2.5_72B_data',
    ]
    candidate_files_ids = [ # The data field has item_id, query, pos, neg
        'misspell2item',
        'negquery2item',
        'query2item',
        'relativequery2item',
        'searchquery2item',
        'title2item',
        'vaguequery2item',
    ]
    candidate_files_i2i = [ # item2item
        'item2item',
    ]
    results = []

    item2freq = defaultdict(int)
    for idx, line in tqdm(enumerate(open(args.in_seq_data)), desc='item freq'):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')
        for itemid in itemids:
            item2freq[int(itemid)] += 1
    # items = list(item2freq.keys())
    # weights = list(item2freq.values())
    # sorted_item2freq = sorted(item2freq.items(), key=lambda x: x[1], reverse=True)

    itemid2text, itemid2title, itemid2features, itemid2price_date_map = get_item_text(args.in_meta_data)

    for file in args.candidate_files.split(','):
        infix = file.split('/')[-1].split('.jsonl')[0]
        item2samples = defaultdict(list)
        total_data = 0
        with open(file, 'r') as f:
            for line in tqdm(f, desc=f'read {file}'):
                data = json.loads(line)
                total_data += 1
                if infix in candidate_files_ids:
                    if isinstance(data['item_id'], list):
                        for k, item in enumerate(data['item_id']):
                            new_data = deepcopy(data)
                            new_data['item_id'] = item
                            new_data['pos'] = [new_data['pos'][k]]
                            item2samples[item].append(new_data)
                    else:
                        item2samples[data['item_id']].append(data)
                elif infix in candidate_files_query:
                    if len(data['pos'])>1:
                        for k, pos in enumerate(data['pos']):
                            new_data = deepcopy(data)
                            new_data['pos'] = [pos]
                            idx = itemid2text.index(new_data['pos'][0])
                            item2samples[idx].append(new_data)
                    else:
                        idx = itemid2text.index(data['pos'][0])
                        item2samples[idx].append(data)
                elif infix in candidate_files_i2i:
                    item2samples[data['pos_id']].append(data)
                else:
                    raise ValueError(f'unknown file {file}')
        
        all_items = list(item2samples.keys())
        all_weights = [item2freq[item] for item in all_items]
        cur_count = 0
        pbar = tqdm(desc=f"sample {file}", total=total_data//2)
        while True:
            sampled_item = random.choices(all_items, weights=all_weights, k=1)[0]
            cur_data = random.choice(item2samples[sampled_item])
            results.append(cur_data)
            cur_count += 1
            pbar.update(1)
            if cur_count >= total_data//2:
                break

    print(f"len(results): {len(results)}")
    with open(args.output_file, 'w') as f:
        for data in results:
            f.write(json.dumps(data)+'\n')