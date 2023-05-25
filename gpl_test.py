import os
import shutil
import json
import csv

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import evaluate
import gpl
from data import *



def make_gpl_generated_directory(dir, df, reverse_queries_and_content, qgen_prefix="explanations"):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=False)

    if reverse_queries_and_content:
        get_corpus_id   = lambda r: f"expl-{r['id']}"
        get_corpus_text = lambda r: f"expl-{r['explanation']}"
        get_query_id    = lambda r: f"row-{r['id']}"
        get_query_text  = lambda r: f"row-{r['arrow_sentence']}"
    else:
        get_corpus_id   = lambda r: f"row-{r['id']}"
        get_corpus_text = lambda r: f"row-{r['arrow_sentence']}"
        get_query_id    = lambda r: f"expl-{r['id']}"
        get_query_text  = lambda r: f"expl-{r['explanation']}"

    with open(f"{dir}/corpus.jsonl", "w") as f:
        for _, row in df.iterrows():
            entry = {
                "_id": get_corpus_id(row),
                "title": "",
                "text": get_corpus_text(row),
            }
            json.dump(entry, f)
            f.write("\n")

    corpus_id_to_query_id = {}
    with open(f"{dir}/{qgen_prefix}-queries.jsonl", "w") as f:
        for _, row in df.iterrows():
            id = get_query_id(row)
            corpus_id_to_query_id[get_corpus_id(row)] = id

            entry = {
                "_id": id,
                "metadata": {},
                "text": get_query_text(row),
            }
            json.dump(entry, f)
            f.write("\n")

    relations_dir = f"{dir}/{qgen_prefix}-qrels"
    os.makedirs(relations_dir, exist_ok=False)

    with open(f"{relations_dir}/train.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        
        for _, row in df.iterrows():
            corpus_id = get_corpus_id(row)
            writer.writerow([corpus_id_to_query_id[corpus_id], corpus_id, 1])

def prepare_gpl_workingdir(base_dir, train_df, val_df, reverse_queries_and_content):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=False)

    make_gpl_generated_directory(f"{base_dir}/train_generated", train_df, reverse_queries_and_content)
    #make_gpl_generated_directory(f"{base_dir}/valid_generated", val_df, reverse_queries_and_content)

def train_gpl(dir, base_ckpt):
    gpl.train(
        path_to_generated_data=f"{dir}/train_generated",
        output_dir=f"{dir}/output",
        qgen_prefix="explanations",
        do_evaluation=False,
        
        base_ckpt=base_ckpt,  
        #base_ckpt='GPL/msmarco-distilbert-margin-mse',

        gpl_score_function="dot",
        batch_size_gpl=16,
        gpl_steps=140000,
        new_size=-1,
        queries_per_passage=-1,
        
        retrievers=               ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        retriever_score_functions=["cos_sim",                    "cos_sim"],
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        
        use_amp=True,
        generator="BeIR/query-gen-msmarco-t5-base-v1", # not important
    )