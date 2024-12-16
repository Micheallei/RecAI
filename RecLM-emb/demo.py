# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
import random
from collections import defaultdict

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from preprocess.utils import get_item_text
from src.huggingface_model_infer import run_model_embedding
from src.openai_model_infer import run_api_embedding
import gradio as gr

def gen_retrieval_result(item2freq, item_embedding_prompt_path, topk, item_embedding_path, user_embedding_path):
    itemid2title = []
    itemid2features = []
    with open(item_embedding_prompt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            itemid2title.append(line['title'])
            itemid2features.append(line['features'])

    item_embeddings = torch.tensor(pickle.load(open(item_embedding_path, "rb")))
    user_embeddings = torch.tensor(pickle.load(open(user_embedding_path, "rb")))
    print("shape of item embeddings: ", item_embeddings.shape)
    print("shape of user embeddings: ", user_embeddings.shape)

    output_text = ""
    output_json = []
    for user_emb in user_embeddings:
        scores = torch.softmax(torch.matmul(user_emb, item_embeddings.T), -1).squeeze().tolist()
        scores = [(index, score) for index, score in enumerate(scores) if index!=0]
        top_itemids = sorted(scores, key=lambda x:-x[1])[:topk]
        # data = {
        #     "result": [(x[0], itemid2title[x[0]][1]) for x in top_itemids]
        # }
        for i, x in enumerate(top_itemids):
            output_text += f"{i+1}.    {itemid2title[x[0]][1]}\n"
            json_dict = {
                "title": itemid2title[x[0]][1],
                "id": x[0],
                "frequency": item2freq[x[0]],
            }
            for elem in itemid2features[x[0]]:
                json_dict[elem[0][:-2]] = elem[1]
            output_json.append(json_dict)
        output_text += "\n"
    return output_text, output_json
        

def parse_args():
    parser = argparse.ArgumentParser(description="infer case")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--user_embedding_prompt_path", type=str, help="Path to query file"
    )
    parser.add_argument(
        "--model_path_or_name", type=str, help="Path to pretrained model or model identifier from huggingface.co/models"
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator()

    item2freq = defaultdict(int)
    for idx, line in tqdm(enumerate(open(args.in_seq_data)), desc='item freq'):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')
        for itemid in itemids:
            item2freq[int(itemid)] += 1

    ## get the dir of args.user_embedding_prompt_path
    cache_dir =  os.path.dirname(args.user_embedding_prompt_path)
    item_embedding_prompt_path = os.path.join(cache_dir, 'item_embedding_prompt.jsonl')
    our_model_item_embedding_path = os.path.join(cache_dir, 'item_embedding.pkl')
    e5_model_item_embedding_path = os.path.join(cache_dir, 'item_embedding_e5.pkl')
    text_3_large_model_item_embedding_path = os.path.join(cache_dir, 'item_embedding_text_3_large.pkl')
    bge_model_item_embedding_path = os.path.join(cache_dir, 'item_embedding_bge.pkl')
    user_embedding_path = os.path.join(cache_dir, 'user_embedding.pkl')

    if not os.path.exists(our_model_item_embedding_path):
        if accelerator.is_main_process:
            os.makedirs(cache_dir, exist_ok=True)
            get_item_text(args.in_meta_data, save_item_prompt_path=item_embedding_prompt_path)
        accelerator.wait_for_everyone()

        print("infer item embedding")
        run_model_embedding(args.model_path_or_name, max_seq_len=args.passage_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=item_embedding_prompt_path, emb_path=our_model_item_embedding_path, accelerator=accelerator, args=args, qorp='passage')
        run_model_embedding("intfloat/e5-large-v2", max_seq_len=args.passage_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=item_embedding_prompt_path, emb_path=e5_model_item_embedding_path, accelerator=accelerator, args=args, qorp='passage')
        run_api_embedding("text-embedding-3-large", item_embedding_prompt_path, text_3_large_model_item_embedding_path)
        run_model_embedding("BAAI/bge-large-en-v1.5", max_seq_len=args.passage_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=item_embedding_prompt_path, emb_path=bge_model_item_embedding_path, accelerator=accelerator, args=args, qorp='passage', sentence_pooling_method='cls')

    def user_infer(text, model_choice, topk):
        print("infer user embedding")
        with open(args.user_embedding_prompt_path, "w", encoding='utf-8') as f:
            f.write(json.dumps({"text": text})+'\n')
        
        if model_choice == "intfloat/e5-large-v2":
            run_model_embedding(model_choice, max_seq_len=args.query_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=args.user_embedding_prompt_path, emb_path=user_embedding_path, accelerator=accelerator, args=args, qorp='query')
            if accelerator.is_main_process:
                base_model_output = gen_retrieval_result(item2freq, item_embedding_prompt_path, topk, e5_model_item_embedding_path, user_embedding_path)
            accelerator.wait_for_everyone()
        elif model_choice == "text-embedding-3-large":
            run_api_embedding(model_choice, args.user_embedding_prompt_path, user_embedding_path)
            if accelerator.is_main_process:
                base_model_output = gen_retrieval_result(item2freq, item_embedding_prompt_path, topk, text_3_large_model_item_embedding_path, user_embedding_path)
            accelerator.wait_for_everyone()
        elif model_choice == "BAAI/bge-large-en-v1.5":
            run_model_embedding(model_choice, max_seq_len=args.query_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=args.user_embedding_prompt_path, emb_path=user_embedding_path, accelerator=accelerator, args=args, qorp='query', sentence_pooling_method='cls')
            if accelerator.is_main_process:
                base_model_output = gen_retrieval_result(item2freq, item_embedding_prompt_path, topk, bge_model_item_embedding_path, user_embedding_path)
            accelerator.wait_for_everyone()
        
        run_model_embedding(args.model_path_or_name, max_seq_len=args.query_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=args.user_embedding_prompt_path, emb_path=user_embedding_path, accelerator=accelerator, args=args, qorp='query')
        if accelerator.is_main_process:
            our_model_output = gen_retrieval_result(item2freq, item_embedding_prompt_path, topk, our_model_item_embedding_path, user_embedding_path)
        accelerator.wait_for_everyone()
        return base_model_output[0], base_model_output[1], our_model_output[0], our_model_output[1]

    def clear():
        return "", {}, "", {}

    with gr.Blocks() as demo:
        gr.HTML("<h1>RecLM-emb Demo</h1>")
        gr.HTML("<p>This is a demo for paper: <a href='https://dl.acm.org/doi/10.1145/3589335.3651468'>Aligning Language Models for Versatile Text-based Item Retrieval</a>.<br><br><b>Domain</b>: xbox games</p>")
        
        with gr.Row():
            model_selection = gr.Dropdown(
                label="Base Model",
                choices=["intfloat/e5-large-v2", "text-embedding-3-large", "BAAI/bge-large-en-v1.5"],
                value="intfloat/e5-large-v2"
            )
            topk_selection = gr.Number(label="Top K", value=10)
        with gr.Row():
            with gr.Column():
                input_text = gr.TextArea(label="User Query", placeholder="Type your query here")
                submit_button = gr.Button("Submit")
                clear_button = gr.ClearButton(components=[input_text], value="Clear")
            
            with gr.Column():
                model_1_output = gr.TextArea(label="Base Model Output") 
                with gr.Accordion("See Details", open=False):
                    output_json_1 = gr.JSON(label="Item Metadata For Base Model") 

            with gr.Column():
                model_2_output = gr.TextArea(label="Our Model Output")
                with gr.Accordion("See Details", open=False):
                    output_json_2 = gr.JSON(label="Item Metadata For Our Model")

        examples = [
                    ["I'd like to find some shooting games that are not made for kids and not 2D platformers.", "intfloat/e5-large-v2", 10],
                    ["I'm looking for a sports game, with high quality graphics and soundtrack, released after 2021.", "intfloat/e5-large-v2", 10],
                    ["The Ascend", "intfloat/e5-large-v2", 10],
                    ["Search for games with exploration and science fiction tags, released before 2015.", "text-embedding-3-large", 10],
                    ["Recommend games excluding these elements: Survival Horror and Fighting.", "text-embedding-3-large", 10],
                    ["Search for games with exploration and science fiction tags, developed by M2.", "text-embedding-3-large", 10],
                    ["I'd like to find some shooting games that are not made for kids and not 2D platformers.", "BAAI/bge-large-en-v1.5", 10],
                    ["I'm looking for a sports game, with high quality graphics and soundtrack, released after 2021.", "BAAI/bge-large-en-v1.5", 10],
                    ["Call of Duty", "BAAI/bge-large-en-v1.5", 10],
                    ["strategy games", "BAAI/bge-large-en-v1.5", 10],
                    ["I like playing sports games, what do you recommend?", "BAAI/bge-large-en-v1.5", 10],
                    # "User: Hi, can you recommend me a game based on what I have played before?\nAssistant: Of course! Please tell me some games you have enjoyed playing.\nUser: I loved Call of Duty Black Ops Cold War, Rust Console Edition, Rocket League, Minecraft, and Fortnite.\nAssistant: Thanks for sharing. What are you looking for in a new game?\nUser: I want a 1st person shooter with a great story, multiplayer, and high-quality soundtrack.",
                    # "User: Hey, I'm looking for a new game recommendation. I've played Minecraft, State of Decay 2, Rocket League, For Honor, Middle-earth Shadow of War, Call of Duty Modern Warfare II, and Overwatch 2.\nAssistant: Thanks for sharing your gaming history. What type of game are you looking for now? Any specific genre or features?\nUser: I want something with 3D graphics, realistic art, and a high-quality soundtrack. I also like combat games and multiplayer options.\nAssistant: Got it. Do you prefer a specific setting or any additional features in the game?\nUser: I have a limited budget of $100.",
                ]
        gr.Examples(examples=examples, inputs=[input_text, model_selection, topk_selection], outputs=[model_1_output, output_json_1, model_2_output, output_json_2], fn=user_infer, cache_examples=True)

        submit_button.click(fn=user_infer, inputs=[input_text, model_selection, topk_selection], outputs=[model_1_output, output_json_1, model_2_output, output_json_2])
        clear_button.click(fn=clear, outputs=[model_1_output, output_json_1, model_2_output, output_json_2])
    
    demo.launch(share=True) 
    