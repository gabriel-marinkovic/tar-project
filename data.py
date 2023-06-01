import logging
import json
from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import iqr
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)



def make_base_dataset():
    dataset = load_dataset("humicroedit", "subtask-1")

    train_df = pd.DataFrame(dataset["train"])
    val_df   = pd.DataFrame(dataset["validation"])
    test_df  = pd.DataFrame(dataset["test"])

    dfs = [train_df, test_df, val_df]
    for df in dfs:
        df["normalized_score"]      = df["meanGrade"] / 3.0
        df["all_scores"]            = df["grades"].apply(lambda s: np.array(sorted([int(c) for c in s])))
        df["all_scores_normalized"] = df["all_scores"].apply(lambda s: s / 3)

        def edit_the_headline(original, edit):
            openIdx  = original.index("<")
            closeIdx = original.index("/>") + len("/>")
            return original[:openIdx] + edit + original[closeIdx:]
        
        df["original_sentence"] = df["original"].apply(lambda s: s.replace("<", "").replace("/>", ""))
        df["edited_sentence"]   = df.apply(lambda row: edit_the_headline(row["original"], row["edit"]), axis=1)

        df["original_word_start_idx"] = df["original"].apply(lambda s: s.index("<"))        
        df["original_word_end_idx"]   = df["original"].apply(lambda s: s.index("/>") - 1)

        df["edited_word_start_idx"] = df["original"].apply(lambda s: s.index("<"))
        df["edited_word_end_idx"]   = df.apply(lambda row: row["edited_word_start_idx"] + len(row["edit"]), axis=1)

        df["arrow_sentence"] = df.apply(
            lambda row: row["original"].replace("<", "[ ").replace("/>", f" => {row['edit']} ]"),
            axis=1
        )

        # statistics thingies
        df['grades'] = df['all_scores']
        df['stddev'] = df['grades'].apply(np.std)
        df['iqr']    = df['grades'].apply(iqr)

        # make grades_max_5 column, which takes the 5 quantile of the grades
        df['grades_max_5'] = df['grades'].apply(lambda x: np.quantile(sorted(x), [0.0, 0.25, 0.5, 0.75, 1.0], method='nearest'))

        def disagreements_fn(row):
            grades = list(row['grades_max_5'])
            count = Counter(grades)
            most_common_grade, freq = count.most_common(1)[0]
            num_disagreements = len(grades) - freq
            return num_disagreements, most_common_grade if freq > 1 else None

        df['disagreements'], df['most_common_grade'] = zip(*df.apply(disagreements_fn, axis=1))

    return train_df, val_df, test_df



def headline_key(original, edit):
    return f"{original} $$$ {edit}"

def load_explanations(df, explanations_path, drop_without_explanation=True):
    with open(explanations_path, "r") as f:
        explanations = json.load(f)

    def get_explanation(r):
        key = headline_key(r["original"], r["edit"])
        expls = explanations.get(key, [])
        return expls[0] if len(expls) > 0 else ""

    df["explanation"] = df.apply(get_explanation, axis=1)
    if drop_without_explanation:
        df = df[df["explanation"] != ""]

    return df



class SentencesDataset(Dataset):
    def __init__(self, dataframe, column_name):
        self.df = dataframe
        self.texts = self.df[column_name].tolist()
        self.scores = torch.Tensor(self.df["normalized_score"].tolist())

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.scores[idx]

class TokenizedSentencesDataset(Dataset):
    def __init__(self, tokenizer, dataframe, column_name, score_dims=1, device="cpu"):
        self.df = dataframe
        self.score_dims = score_dims

        #self.normalized_scores = torch.Tensor(self.df["most_common_grade"].to_numpy() / 3).to(device)
        self.normalized_scores = torch.Tensor(self.df["normalized_score"].to_numpy()).to(device)

        original = self.df["original_sentence"].tolist()
        edited   = self.df["edited_sentence"].tolist()
        text = [f"{o} [SEP] {e}" for o, e in zip(original, edited)]
        #text = self.df[column_name].tolist()

        output = tokenizer(text=text, truncation=True, padding=True, return_tensors='pt').to(device)
        
        self.input_ids      = output["input_ids"]
        self.attention_mask = output["attention_mask"]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        labels = self.normalized_scores[idx]
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         labels,
        }
    
class TokenizedSentencePairsDataset(Dataset):
    def __init__(self, tokenizer, dataframe, column1, column2, device="cpu"):
        self.df = dataframe
        
        self.normalized_scores = torch.Tensor(self.df["normalized_score"].tolist()).to(device)

        score_counts = torch.zeros(len(self.df.index), 4).to(device)
        for i, scores in self.df["all_scores"].items():
            for score in scores:
                score_counts[i, score] += 1
        self.score_counts = score_counts.to(device)

        pairs = (
            self.df[column1].tolist(),
            self.df[column2].tolist(),
        )
        output = tokenizer(text_pairs=pairs, truncation=True, padding=True, return_tensors='pt').to(device)
        
        self.input_ids      = output["input_ids"]
        self.attention_mask = output["attention_mask"]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.normalized_scores[idx],
            #"labels":         self.score_counts[idx],
        }
    
class TokenizedSentencesForMLMDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], 
                                    add_special_tokens=True, 
                                    truncation=True, 
                                    max_length=self.max_length, 
                                    return_special_tokens_mask=True)
        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], 
                                                    add_special_tokens=True, 
                                                    truncation=True, 
                                                    max_length=self.max_length, 
                                                    return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)