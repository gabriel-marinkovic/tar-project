from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import iqr
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import evaluate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def round_to(n, fraction):
    return round(n / fraction) * fraction

def make_base_dataset():
    dataset = load_dataset("humicroedit", "subtask-1")

    train_df = pd.DataFrame(dataset["train"])
    val_df   = pd.DataFrame(dataset["validation"])
    test_df  = pd.DataFrame(dataset["test"])

    dfs = [train_df, test_df, val_df]
    for df in dfs:
        df['mean_grade_bucket']     = df['meanGrade'].apply(lambda x: round_to(x, 0.5))
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



mse_metric = evaluate.load("mse")
def compute_metrics_regression(eval_pred):
    predictions, labels = eval_pred
    predictions, labels = torch.tensor(predictions).squeeze(), torch.tensor(labels).squeeze()
    predictions *= 3
    labels      *= 3
    
    mse = mse_metric.compute(predictions=predictions, references=labels)
    return { "mse":  mse["mse"], "rmse": np.sqrt(mse["mse"]) }



class RegressionDataset(Dataset):
    def __init__(self, tokenizer, dataframe):
        self.df = dataframe
        #self.labels = torch.Tensor(self.df["most_common_grade"].to_numpy() / 3)
        self.labels = torch.Tensor(self.df["normalized_score"].to_numpy())

        original = self.df["original_sentence"].tolist()
        edited   = self.df["edited_sentence"].tolist()
        text = [f"{e}[SEP]{o}" for o, e in zip(original, edited)]
        output = tokenizer(text=text, is_split_into_words=False, truncation=True, padding=True, return_tensors='pt')
        
        self.input_ids      = output["input_ids"]
        self.attention_mask = output["attention_mask"]

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

class UnlabeledDataset(Dataset):
    def __init__(self, tokenizer, dataframe):
        self.df = dataframe
        
        original = self.df["original_sentence"].tolist()
        edited   = self.df["edited_sentence"].tolist()
        text = [f"{e}[SEP]{o}" for o, e in zip(original, edited)]
        #text = edited
        output = tokenizer(text=text, truncation=True, padding=True, return_tensors='pt')
        
        self.input_ids      = output["input_ids"]
        self.attention_mask = output["attention_mask"]
    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
    

def fit_regression(base_model_dir, save_dir, train_df, val_df, test_df, num_epochs=5):
    model     = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    
    args = TrainingArguments(
        save_dir,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=100,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        learning_rate=5e-5,
        #weight_decay=0.001,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        disable_tqdm=True,
        
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=RegressionDataset(tokenizer, train_df),
        eval_dataset=RegressionDataset(tokenizer, val_df),
        compute_metrics=compute_metrics_regression,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(RegressionDataset(tokenizer, test_df))
    test_metrics = compute_metrics_regression((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return trainer

def load_regression_trainer(dir):
    model = AutoModelForSequenceClassification.from_pretrained(dir, num_labels=1)
    return Trainer(model)


def train_and_evaluate_model(data_section, make_baseline_model, make_clean_model):
    underlying_model = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(underlying_model)
    train_df, val_df, test_df = make_base_dataset()

    def cleanup_df(df):
        df = df.copy()
        if data_section == 1:
            df = df[df["meanGrade"] >= 0.4] # remove Q1
        elif data_section == 2:
            df = df[(df["meanGrade"] <= 0.4) | (df["meanGrade"] >= 1.4)] # remove Q1-Q3
        elif data_section == 3:
            df = df[df["meanGrade"] <= 1.4] # remove Q3
        else:
            assert False and "invalid data_section"
        return df
    
    clean_train_df = cleanup_df(train_df)
    clean_val_df   = cleanup_df(val_df)
    clean_test_df  = cleanup_df(test_df)
    

    if make_baseline_model:
        print("Fitting baseline...")
        trainer_baseline = fit_regression(underlying_model, "./models/baseline", train_df, val_df, test_df)
    else:
        print("Loading baseline...")
        trainer_baseline = load_regression_trainer("./models/baseline")

    if make_clean_model:
        print("Fitting clean...")
        trainer_clean = fit_regression(underlying_model, "./models/clean", clean_train_df, clean_val_df, clean_test_df)
    else:
        print("Loading clean...")
        trainer_clean = load_regression_trainer("./models/clean")


    out_baseline = trainer_baseline.predict(RegressionDataset(tokenizer, test_df))
    baseline_regression_score = out_baseline.predictions.squeeze()
    cleaned_regression_score = trainer_clean.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze()

    df = test_df.copy()
    df["baseline_regression_score"] = baseline_regression_score
    df["cleaned_regression_score"]  = cleaned_regression_score

    path = f"./df_subset_{data_section}.pkl"
    df.to_pickle(path)
    

if __name__ == "__main__":
    print("Doing section 1...")
    train_and_evaluate_model(data_section=1, make_baseline_model=False, make_clean_model=True) # remove below Q1
    print("Doing section 2...")
    train_and_evaluate_model(data_section=2, make_baseline_model=False, make_clean_model=True) # remove between Q1 and Q3
    print("Doing section 3...")
    train_and_evaluate_model(data_section=3, make_baseline_model=False, make_clean_model=True) # remove above Q3