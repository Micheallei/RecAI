import mteb
from sentence_transformers import SentenceTransformer, models
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--model_names_or_path", type=str, help="", required=True
    )
    parser.add_argument(
        "--tasks", type=str, help="", default="ArguAna,NFCorpus,BiorxivClusteringS2S,SciDocsRR"
    )
    parser.add_argument(
        "--output_path", type=str, help=""
    )
    parser.add_argument(
        "--batch_size", type=int, help="", default=1024
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    word_embedding_model = models.Transformer(args.model_names_or_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.sentence_pooling_method)
    prompts = {"query": "query: ", "passage": "passage: "} if args.has_template else None
    if args.normlized:
        normalized_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalized_layer], prompts=prompts)
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], prompts=prompts)

    tasks = mteb.get_tasks(tasks=args.tasks.split(","))
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.output_path, encode_kwargs = {'batch_size': args.batch_size})