import json
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate

from data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    



BCE_BIAS = 0.8

def compute_metrics_bce(eval_pred):
    def maybe(fn):
        try: return fn()
        except: return 0.0

    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).squeeze().double()
    labels      = torch.tensor(labels)     .squeeze().double()

    bce = torch.nn.BCEWithLogitsLoss()(predictions, labels).item()

    predictions = torch.sigmoid(predictions) > BCE_BIAS

    accuracy    = maybe(lambda: torch.sum(predictions == labels).item() / len(labels))
    precision   = maybe(lambda: torch.sum(predictions * labels).item() / torch.sum(predictions).item())
    recall      = maybe(lambda: torch.sum(predictions * labels).item() / torch.sum(labels).item())
    f1          = maybe(lambda: 2 * precision * recall / (precision + recall))

    return {
        "bce": bce,
        "acc": accuracy,
        "p": precision,
        "r": recall,
        "f1": f1,
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

class BCEDataset(Dataset):
    def __init__(self, tokenizer, dataframe, bool_column):
        self.df = dataframe
        
        original = self.df["original_sentence"].tolist()
        edited   = self.df["edited_sentence"].tolist()
        text = [f"{e}[SEP]{o}" for o, e in zip(original, edited)]
        #text = edited
        output = tokenizer(text=text, truncation=True, padding=True, return_tensors='pt')
        
        self.input_ids      = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels         = torch.tensor(self.df[bool_column].values, dtype=torch.float32)
    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

class BCETrainer(Trainer):
    def __init__(self, positive_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_weight = positive_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits").squeeze()
        if len(logits.shape) == 0:
            logits = logits.unsqueeze(0)

        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.positive_weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def fit_model_bce(base_model_dir, save_dir, train_df, val_df, test_df, bool_column):
    model     = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    args = TrainingArguments(
        save_dir,
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        num_train_epochs=10,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        weight_decay=0.001,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    pos_count = train_df[bool_column].sum()
    neg_count = len(train_df.index) - pos_count
    pos_weight = torch.tensor(neg_count / pos_count).to(device)
    print("positive weight is", pos_weight)
    if pos_count > neg_count:
        print("WARN pos_count > neg_count, you should probably switch labels")
    
    trainer = BCETrainer(
        pos_weight,

        model,
        args,
        train_dataset=BCEDataset(tokenizer, train_df, bool_column),
        eval_dataset=BCEDataset(tokenizer, val_df, bool_column),
        compute_metrics=compute_metrics_bce,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)]
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(BCEDataset(tokenizer, test_df, bool_column))
    test_metrics = compute_metrics_bce((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return trainer

def load_bce_trainer(dir):
    model = AutoModelForSequenceClassification.from_pretrained(dir, num_labels=1)
    return BCETrainer(0, model)


def is_row_terrible(row):
    #if row["most_common_grade"] == 0 and row["meanGrade"] < 1:
    #    return 1
    #elif row["most_common_grade"] == 1 and row["meanGrade"] <= 1:
    #    return 1
    if row["meanGrade"] <= 0.55:
        return 1
    else:
        return 0

def is_row_decent(row):
    if row["meanGrade"] >= 1.5:
        return 1
    else:
        return 0

if False:
    dfs = [train_df, val_df, test_df]
    for i in range(len(dfs)):
        #df0 = dfs[i][dfs[i]["disagreements"] == 0]
        #df1 = dfs[i][dfs[i]["disagreements"] == 1]
        #df2 = dfs[i][dfs[i]["disagreements"] == 2]
        #dfs[i] = pd.concat([df0, df1, df2], ignore_index=True)
        dfs[i]["is_terrible"] = dfs[i].apply(is_row_terrible, axis=1)
        dfs[i]["is_decent"]   = dfs[i].apply(is_row_decent,   axis=1)

        # keep only is_terrible and is_decent rows
        dfs[i] = dfs[i][dfs[i]["is_terrible"] + dfs[i]["is_decent"] > 0]
    train_df, val_df, test_df = dfs

def make_preprocessed_dataset(remove_limp_examples):
    dfs = list(make_base_dataset())

    for i in range(len(dfs)):
        dfs[i]["is_poor"]   = dfs[i].apply(is_row_terrible, axis=1)
        dfs[i]["is_decent"] = dfs[i].apply(is_row_decent,   axis=1)

        # keep only is_terrible and is_decent rows
        if remove_limp_examples:
            dfs[i] = dfs[i][dfs[i]["is_poor"] + dfs[i]["is_decent"] > 0]

    return dfs

def lerp(a, b, t):
    return a + t * (b - a)

def do_thing_that_kinda_works():
    #underlying_model = 'bert-base-uncased'
    #underlying_model = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    underlying_model = 'roberta-base'
    
    tokenizer = AutoTokenizer.from_pretrained(underlying_model)

    make_clean_discriminator = True
    discriminate_datasets = True
    make_baseline_model = False
    make_refined_model = True

    if make_clean_discriminator:
        print("Fitting discriminator...")
        train_df_reduced, val_df_reduced, test_df_reduced = make_preprocessed_dataset(remove_limp_examples=True)
        discriminator = fit_model_bce(underlying_model, "./models/discriminator_poor",  train_df_reduced, val_df_reduced, test_df_reduced, "is_poor")    
    else:
        print("Loading discriminator...")
        discriminator = load_bce_trainer("./models/discriminator_poor")

    if discriminate_datasets:
        print("Discriminating datasets...")
        train_df, val_df, test_df = make_base_dataset()

        def discriminate_df(df):
            df["is_poor"] = 0
            df["poor_score"] = 0
            for i in range(0, len(df.index), 1000):
                print(i)
                preds = discriminator.predict(UnlabeledDataset(tokenizer, df[i:i+1000])).predictions
                preds = torch.tensor(preds).squeeze()
                if len(preds.shape) == 0:
                    preds = preds.unsqueeze(0)

                preds          = torch.sigmoid(preds)
                preds_discrete = preds > BCE_BIAS
                df.loc[i:i+1000-1, "is_poor"]    = preds_discrete.numpy()
                df.loc[i:i+1000-1, "poor_score"] = preds.numpy()
            
            return df
        train_df = discriminate_df(train_df)
        val_df   = discriminate_df(val_df)
        test_df  = discriminate_df(test_df)

        train_df.to_pickle("train_df_discriminated.pkl")
        val_df.to_pickle("val_df_discriminated.pkl")
        test_df.to_pickle("test_df_discriminated.pkl")
    else:
        print("Loading discriminated datasets...")
        train_df = pd.read_pickle("train_df_discriminated.pkl")
        val_df   = pd.read_pickle("val_df_discriminated.pkl")
        test_df  = pd.read_pickle("test_df_discriminated.pkl")

    if make_baseline_model:
        print("Fitting baseline...")

        #train_df_please = train_df[(train_df['mean_grade_bucket'] != 1.0) & (train_df['mean_grade_bucket'] != 1.5)]
        #val_df_please   = val_df[(val_df['mean_grade_bucket'] != 1.0) & (val_df['mean_grade_bucket'] != 1.5)]
        #test_df_please  = test_df[(test_df['mean_grade_bucket'] != 1.0) & (test_df['mean_grade_bucket'] != 1.5)]

        trainer_baseline = fit_regression(underlying_model, "./models/baseline", *make_base_dataset())
    else:
        print("Loading baseline...")
        trainer_baseline = load_regression_trainer("./models/baseline")

    
    train_df_poor     = train_df[train_df["is_poor"] == 1]
    train_df_not_poor = train_df[train_df["is_poor"] == 0]    
    val_df_poor       = val_df[val_df["is_poor"] == 1]
    val_df_not_poor   = val_df[val_df["is_poor"] == 0]
    test_df_poor      = test_df[test_df["is_poor"] == 1]
    test_df_not_poor  = test_df[test_df["is_poor"] == 0]

    if make_refined_model:
        print("Fitting refined model...")

        trainer_decent = fit_regression(underlying_model, "./models/refined_decent", train_df_not_poor, val_df_not_poor, test_df_not_poor, num_epochs=10)

        # UNCOMMENT FOR SOME RESULTS
        # it seems fine to train classifier 2 on everything that isnt in meangradebucket == 1, 
        # we get some results when taking into account stddev
        #
        #train_df_please = train_df[(train_df['mean_grade_bucket'] != 1.0)]
        #val_df_please   = val_df[(val_df['mean_grade_bucket'] != 1.0)]
        #test_df_please  = test_df[(test_df['mean_grade_bucket'] != 1.0)]
        #trainer_decent = fit_regression(underlying_model, "./models/refined_decent", train_df_please, val_df_please, test_df_please, num_epochs=10)
    else:
        print("Loading refined model...")
        trainer_decent = load_regression_trainer("./models/refined_decent")

    print("Evaluating...")
    results = []
    
    if False:
        baseline_regresion_score        = trainer_baseline.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze() # nocheckin
        refined_regression_score_poor   = trainer_poor.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze()
        refined_regression_score_decent = trainer_decent.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze()

        test_df["baseline_regresion_score"] = baseline_regresion_score
        test_df["refined_regression_score_poor"] = refined_regression_score_poor
        test_df["refined_regression_score_decent"] = refined_regression_score_decent
        test_df["mixed_regression_score"] = \
            (test_df["is_poor"] * refined_regression_score_poor) + \
            (test_df["is_decent"] * refined_regression_score_decent) + \
            ((1 - test_df["is_poor"]) * (1 - test_df["is_decent"]) * baseline_regresion_score)

        results = []
        for bucket in [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]:
            mean_grade_from = bucket - 0.25
            mean_grade_to   = bucket + 0.25

            test_df_bucket = test_df[(test_df["meanGrade"] >= mean_grade_from) & (test_df["meanGrade"] <= mean_grade_to)]
            if len(test_df_bucket) == 0:
                continue

            baseline_regresion_score_bucket        = test_df_bucket["baseline_regresion_score"].to_numpy().astype(float)
            refined_regression_score_poor_bucket   = test_df_bucket["refined_regression_score_poor"].to_numpy().astype(float)
            refined_regression_score_decent_bucket = test_df_bucket["refined_regression_score_decent"].to_numpy().astype(float)
            mixed_regression_score_bucket          = test_df_bucket["mixed_regression_score"].to_numpy().astype(float)
            mean_grade_bucket                      = test_df_bucket["normalized_score"].to_numpy().astype(float)

            test_metrics_baseline       = compute_metrics_regression((baseline_regresion_score_bucket, mean_grade_bucket))
            test_metrics_refined_poor   = compute_metrics_regression((refined_regression_score_poor_bucket, mean_grade_bucket))
            test_metrics_refined_decent = compute_metrics_regression((refined_regression_score_decent_bucket, mean_grade_bucket))
            test_metrics_mixed          = compute_metrics_regression((mixed_regression_score_bucket,   mean_grade_bucket))

            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, baseline model:      ", test_metrics_baseline)
            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, refined poor model:  ", test_metrics_refined_poor)
            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, refined decent model:", test_metrics_refined_decent)
            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, mixed model:         ", test_metrics_mixed)

            bucket_label = f'{mean_grade_from:.1f} - {mean_grade_to:.1f}'
            results.append({
                "grade_bucket": bucket_label,
                "model": "baseline",
                "rmse": test_metrics_baseline["rmse"],
            })
            results.append({
                "grade_bucket": bucket_label,
                "model": "refined_poor",
                "rmse": test_metrics_refined_poor["rmse"],
            })
            results.append({
                "grade_bucket": bucket_label,
                "model": "refined_decent",
                "rmse": test_metrics_refined_decent["rmse"],
            })
            results.append({
                "grade_bucket": bucket_label,
                "model": "mixed",
                "rmse": test_metrics_mixed["rmse"],
            })
    
    if True:
        baseline_regresion_score = trainer_baseline.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze()
        refined_regression_score = trainer_decent.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze()

        disc_decision = test_df["is_poor"].to_numpy().astype(float)
        disc_score    = test_df["poor_score"].to_numpy()
        disc_confidence_pos = ((disc_score - BCE_BIAS) / (1 - BCE_BIAS)) * 0.5 + 0.5
        disc_confidence_neg = ((BCE_BIAS - disc_score) /      BCE_BIAS)  * 0.5 + 0.5
        
        disc_confidence_pos *= 1 - (disc_decision.sum() / len(disc_decision))
        disc_confidence_neg *=     (disc_decision.sum() / len(disc_decision))

        mixed_regression_score = disc_decision * (
            baseline_regresion_score * disc_confidence_pos + \
            refined_regression_score * (1 - disc_confidence_pos)
        ) + \
        (1 - disc_decision) * (
            baseline_regresion_score * disc_confidence_neg + \
            refined_regression_score * (1 - disc_confidence_neg)
        )

        test_df["baseline_regresion_score"]       = baseline_regresion_score
        test_df["refined_regression_score"]       = refined_regression_score
        test_df["mixed_regression_score_scoring"] = mixed_regression_score
        test_df["mixed_regression_score_lerp"] = lerp(
            baseline_regresion_score, refined_regression_score, 
            np.minimum(baseline_regresion_score * 0.5, 1)
        )
        
        test_df.to_pickle("test_df_normalization_scores.pkl")

    if False:
        baseline_regresion_score = trainer_anti_refined.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze() # nocheckin
        refined_regression_score = trainer_poor.predict(UnlabeledDataset(tokenizer, test_df)).predictions.squeeze()

        disc_decision = test_df["is_poor"].to_numpy().astype(float)
        disc_score    = test_df["discrimination_score"].to_numpy()
        disc_confidence_pos = ((disc_score - BCE_BIAS) / (1 - BCE_BIAS)) * 0.5 + 0.5
        disc_confidence_neg = ((BCE_BIAS - disc_score) /      BCE_BIAS)  * 0.5 + 0.5
        
        disc_confidence_pos *= 1 - (disc_decision.sum() / len(disc_decision))
        disc_confidence_neg *=     (disc_decision.sum() / len(disc_decision))

        mixed_regression_score = disc_decision * (
            baseline_regresion_score * disc_confidence_pos + \
            refined_regression_score * (1 - disc_confidence_pos)
        ) + \
        (1 - disc_decision) * (
            baseline_regresion_score * disc_confidence_neg + \
            refined_regression_score * (1 - disc_confidence_neg)
        )

        test_df["baseline_regresion_score"] = baseline_regresion_score
        test_df["refined_regression_score"] = refined_regression_score
        test_df["mixed_regression_score"]   = mixed_regression_score

        results = []
        for bucket in [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]:
            mean_grade_from = bucket - 0.25
            mean_grade_to   = bucket + 0.25

            test_df_bucket = test_df[(test_df["meanGrade"] >= mean_grade_from) & (test_df["meanGrade"] <= mean_grade_to)]
            if len(test_df_bucket) == 0:
                continue

            baseline_regresion_score_bucket = test_df_bucket["baseline_regresion_score"].to_numpy()
            refined_regression_score_bucket = test_df_bucket["refined_regression_score"].to_numpy()
            mixed_regression_score_bucket   = test_df_bucket["mixed_regression_score"].to_numpy()
            mean_grade_bucket               = test_df_bucket["normalized_score"].to_numpy()

            test_metrics_baseline = compute_metrics_regression((baseline_regresion_score_bucket, mean_grade_bucket))
            test_metrics_refined  = compute_metrics_regression((refined_regression_score_bucket, mean_grade_bucket))
            test_metrics_mixed    = compute_metrics_regression((mixed_regression_score_bucket,   mean_grade_bucket))

            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, baseline model:", test_metrics_baseline)
            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, refined model: ", test_metrics_refined)
            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, mixed model:   ", test_metrics_mixed)

            bucket_label = f'{mean_grade_from:.1f} - {mean_grade_to:.1f}'
            results.append({
                "grade_bucket": bucket_label,
                "model": "baseline",
                "rmse": test_metrics_baseline["rmse"],
            })
            results.append({
                "grade_bucket": bucket_label,
                "model": "refined",
                "rmse": test_metrics_refined["rmse"],
            })
            results.append({
                "grade_bucket": bucket_label,
                "model": "mixed",
                "rmse": test_metrics_mixed["rmse"],
            })

    if False:
        results = []
        for bucket in [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]:
            mean_grade_from = bucket - 0.25
            mean_grade_to   = bucket + 0.25

            test_df_bucket          = test_df         [(test_df         ["meanGrade"] >= mean_grade_from) & (test_df         ["meanGrade"] <= mean_grade_to)]
            test_df_poor_bucket     = test_df_poor    [(test_df_poor    ["meanGrade"] >= mean_grade_from) & (test_df_poor    ["meanGrade"] <= mean_grade_to)]
            test_df_not_poor_bucket = test_df_not_poor[(test_df_not_poor["meanGrade"] >= mean_grade_from) & (test_df_not_poor["meanGrade"] <= mean_grade_to)]
            if len(test_df_bucket) == 0:
                continue
            
            out_baseline = trainer_baseline.predict(RegressionDataset(tokenizer, test_df_bucket))

            out_poor     = { "predictions": np.array([], dtype=float), "label_ids": np.array([], dtype=float) }
            out_not_poor = { "predictions": np.array([], dtype=float), "label_ids": np.array([], dtype=float) }

            if len(test_df_poor_bucket) != 0:
                out_poor = trainer_poor.predict(RegressionDataset(tokenizer, test_df_poor_bucket))._asdict()
                #out_poor = trainer_poor.predict(RegressionDataset(tokenizer, test_df_poor_bucket))._asdict()
                #out_poor = trainer_baseline.predict(RegressionDataset(tokenizer, test_df_poor_bucket))._asdict()
                
            if len(test_df_not_poor_bucket) != 0:
                out_not_poor = trainer_decent.predict(RegressionDataset(tokenizer, test_df_not_poor_bucket))._asdict()

            out_combined_predictions = np.concatenate([out_poor["predictions"].squeeze(), out_not_poor["predictions"].squeeze()])
            out_combined_label_ids   = np.concatenate([out_poor["label_ids"].squeeze(),   out_not_poor["label_ids"].squeeze()])

            test_metrics_baseline = compute_metrics_regression((out_baseline.predictions.squeeze(), out_baseline.label_ids.squeeze()))
            test_metrics_combined = compute_metrics_regression((out_combined_predictions,           out_combined_label_ids))
            
            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, baseline model  :", test_metrics_baseline)
            print(f"ENTIRE dataset, meanGrade {mean_grade_from} - {mean_grade_to}, combined models :", test_metrics_combined)

            bucket_label = f'{mean_grade_from:.1f} - {mean_grade_to:.1f}'
            results.append({
                "grade_bucket": bucket_label,
                "model": "baseline",
                "rmse": test_metrics_baseline["rmse"],
            })
            results.append({
                "grade_bucket": bucket_label,
                "model": "our",
                "rmse": test_metrics_combined["rmse"],
            })

    with open("results_grade_buckets.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Done!")


#do_thing_that_kinda_works()

def new_thing():
    underlying_model = "bert-base-uncased"

    make_baseline_model = False
    make_clean_model = True

    tokenizer = AutoTokenizer.from_pretrained(underlying_model)
    train_df, val_df, test_df = make_base_dataset()

    def cleanup_df(df):
        df = df.copy()
        df = df[df["meanGrade"] <= 1.4] # remove Q3
        #df = df[df["meanGrade"] >= 0.4] # remove Q1
        df = df[(df["meanGrade"] <= 0.4) | (df["meanGrade"] >= 1.4)] # remove Q1-Q3
        #df = df[(df["disagreements"] <= 1)]
        return df
    
    clean_train_df = cleanup_df(train_df)
    clean_val_df   = cleanup_df(val_df)
    clean_test_df  = cleanup_df(test_df)
    
    print("should not be same", len(test_df), len(clean_test_df))

    print("train_df", train_df.groupby("mean_grade_bucket").size())

    if make_baseline_model:
        print("Fitting baseline...")
        sampled_train_df = train_df   #train_df.sample(n=len(clean_train_df))
        sampled_val_df   = val_df     #val_df.sample(n=len(clean_val_df))
        sampled_test_df  = test_df    #test_df.sample(n=len(clean_test_df))
        trainer_baseline = fit_regression(underlying_model, "./models/baseline", sampled_train_df, sampled_val_df, sampled_test_df)
    else:
        print("Loading baseline...")
        trainer_baseline = load_regression_trainer("./models/baseline")

    if make_clean_model:
        print("Fitting clean...")
        trainer_clean = fit_regression(underlying_model, "./models/clean", clean_train_df, clean_val_df, clean_test_df)
    else:
        print("Loading clean...")
        trainer_clean = load_regression_trainer("./models/clean")

    def evaluate_df(df, pickle_path):
        #loaded_df['baseline_regression_score_rmse'] = loaded_df.apply(lambda row: np.sqrt((row['baseline_regression_score']*3 - row["meanGrade"])**2) , axis=1)

        out_baseline = trainer_baseline.predict(RegressionDataset(tokenizer, df))
        baseline_regression_score = out_baseline.predictions.squeeze()
        cleaned_regression_score = trainer_clean.predict(UnlabeledDataset(tokenizer, df)).predictions.squeeze()

        test_metrics_baseline = compute_metrics_regression((out_baseline.predictions.squeeze(), out_baseline.label_ids.squeeze()))
        print("baseline metrics", test_metrics_baseline)

        df = df.copy()
        df["baseline_regression_score"] = baseline_regression_score
        df["cleaned_regression_score"]  = cleaned_regression_score
        df.to_pickle(pickle_path)

    print ("evaluating on full df...")
    evaluate_df(test_df, "./test_df_with_scores3.pkl")

    print ("evaluating on clean df...")
    evaluate_df(clean_test_df, "./clean_test_df_with_scores3.pkl")
    

new_thing()