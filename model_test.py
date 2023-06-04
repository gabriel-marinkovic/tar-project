import logging
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
from gpl_test import *

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def calculate_bucket_accuracy(predicted_scores, actual_scores):
    # Define the boundaries for the buckets
    boundaries = torch.linspace(0, 1, 7 + 1)[1:-1]

    predicted_buckets = torch.bucketize(predicted_scores, boundaries)
    actual_buckets    = torch.bucketize(actual_scores,    boundaries)

    correct_predictions = (predicted_buckets == actual_buckets)

    bucket_accuracy = []
    for i in range(10):
        mask = (actual_buckets == i)
        correct_in_bucket = correct_predictions[mask]
        correct_count = correct_in_bucket.float().sum()
        total_count   = mask.float().sum()
        accuracy = correct_count / total_count
        #accuracy = correct_in_bucket.float().mean().item()
        bucket_accuracy.append(accuracy.item())

    #bucket_accuracy = torch.tensor(bucket_accuracy)
    return bucket_accuracy


mae_metric = evaluate.load("mae")
mse_metric = evaluate.load("mse")
def compute_metrics_regression(eval_pred):
    predictions, labels = eval_pred
    predictions, labels = torch.tensor(predictions).squeeze(), torch.tensor(labels).squeeze()
    predictions *= 3
    labels      *= 3
    
    mse = mse_metric.compute(predictions=predictions, references=labels)
    mae = mae_metric.compute(predictions=predictions, references=labels)
    return {
        "mae":  mae["mae"],
        "mse":  mse["mse"],
        "rmse": np.sqrt(mse["mse"]),
        #"bucket_acc": calculate_bucket_accuracy(predictions / 3, labels / 3),
    }

def compute_metrics_model_annotators(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).squeeze().double()
    labels      = torch.tensor(labels)     .squeeze().double()

    if labels.shape[1] == 4:
        scores = torch.tensor(np.array([0, 1, 2, 3])).double()
    elif labels.shape[1] == 2:
        scores = torch.tensor(np.array([0.5, 2])).double()
    scores = scores.reshape((-1, 1))

    sm = torch.nn.Softmax(dim=1)
    score_predictions = (sm(predictions) @ scores).squeeze()
    score_labels      = (labels          @ scores).squeeze()

    ce = torch.nn.CrossEntropyLoss()(predictions, labels).item()

    mse = mse_metric.compute(predictions=score_predictions, references=score_labels)
    mae = mae_metric.compute(predictions=score_predictions, references=score_labels)
    return {
        "mae":  mae["mae"],
        "mse":  mse["mse"],
        "rmse": np.sqrt(mse["mse"]),
        "cross_entropy": ce,
        #"bucket_acc": calculate_bucket_accuracy(score_predictions / 3, score_labels / 3),
    }
        

def do_masked_language_modeling(base_model_name, save_dir, train_df, val_df, test_df):
    model     = AutoModelForMaskedLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    all_sentences_train = train_df["explanation"].tolist() + train_df["arrow_sentence"].tolist()
    train_dataset = TokenizedSentencesForMLMDataset(all_sentences_train, tokenizer, 256)

    all_sentences_valid = val_df["explanation"].tolist() + val_df["arrow_sentence"].tolist()
    valid_dataset = TokenizedSentencesForMLMDataset(all_sentences_valid, tokenizer, 256, cache_tokenization=True)

    # do whole word masking
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

    training_args = TrainingArguments(
        num_train_epochs=10,
        evaluation_strategy="steps",
        per_device_train_batch_size=64,
        eval_steps=500,
        logging_steps=500,
        prediction_loss_only=True,
        fp16=True,
        output_dir="mlm_output",
        overwrite_output_dir=True,
        save_total_limit=1,
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model


def fit_regression(base_model_dir, save_dir, train_df, val_df, test_df):
    model     = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    
    args = TrainingArguments(
        save_dir,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=100,
        num_train_epochs=5,
        warmup_ratio=0.1,
        learning_rate=5e-6,
        weight_decay=0.001,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=TokenizedSentencesDataset(tokenizer, train_df, "arrow_sentence"),
        eval_dataset=TokenizedSentencesDataset(tokenizer, val_df, "arrow_sentence"),
        compute_metrics=compute_metrics_regression,
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(TokenizedSentencesDataset(tokenizer, test_df, "arrow_sentence"))
    test_metrics = compute_metrics_regression((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model, trainer


def fit_model_annotators(base_model_dir, save_dir, train_df, val_df, test_df):
    model     = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    args = TrainingArguments(
        save_dir,
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        num_train_epochs=20,
        warmup_ratio=0.1,
        learning_rate=5e-7,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="cross_entropy",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=TokenizedSentencesDataset(tokenizer, train_df, "arrow_sentence", score_dims=4),
        eval_dataset=TokenizedSentencesDataset(tokenizer, val_df, "arrow_sentence", score_dims=4),
        compute_metrics=compute_metrics_model_annotators,
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(TokenizedSentencesDataset(tokenizer, test_df, "arrow_sentence", score_dims=4))
    test_metrics = compute_metrics_model_annotators((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model

def fit_model_annotators_binary(base_model_dir, save_dir, train_df, val_df, test_df):
    model     = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    args = TrainingArguments(
        save_dir,
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        num_train_epochs=20,
        warmup_ratio=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="cross_entropy",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=TokenizedSentencesDataset(tokenizer, train_df, "arrow_sentence", score_dims=2),
        eval_dataset=TokenizedSentencesDataset(tokenizer, val_df, "arrow_sentence", score_dims=2),
        compute_metrics=compute_metrics_model_annotators,
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(TokenizedSentencesDataset(tokenizer, test_df, "arrow_sentence", score_dims=2))
    test_metrics = compute_metrics_model_annotators((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model

def fit_model_detect_decent(base_model_dir, save_dir, train_df, val_df, test_df):
    def is_terrible(row):
        if row["most_common_grade"] == 0 and row["disagreements"] <= 2:
            return 1
        else:
            return 0
    def is_decent(row):
        if row["most_common_grade"] >= 2:
            return 1
        else:
            return 0
        
    dfs = [train_df, val_df, test_df]
    for i in range(len(dfs)):
        #df0 = dfs[i][dfs[i]["disagreements"] == 0]
        #df1 = dfs[i][dfs[i]["disagreements"] == 1]
        #df2 = dfs[i][dfs[i]["disagreements"] == 2]
        #dfs[i] = pd.concat([df0, df1, df2], ignore_index=True)
        dfs[i]["is_terrible"] = dfs[i].apply(is_terrible, axis=1)
        dfs[i]["is_decent"]   = dfs[i].apply(is_decent,   axis=1)

        # keep only is_terrible and is_decent rows
        dfs[i] = dfs[i][dfs[i]["is_terrible"] + dfs[i]["is_decent"] > 0]
    train_df, val_df, test_df = dfs

    class TerribleDataset(Dataset):
        def __init__(self, tokenizer, dataframe):
            self.df = dataframe
            
            original = self.df["original_sentence"].tolist()
            edited   = self.df["edited_sentence"].tolist()
            #text = [f"{o} [SEP] {e}" for o, e in zip(original, edited)]
            text = edited
            output = tokenizer(text=text, truncation=True, padding=True, return_tensors='pt')
            
            self.input_ids      = output["input_ids"]
            self.attention_mask = output["attention_mask"]
            self.labels         = torch.tensor(self.df['is_decent'].values, dtype=torch.float32)
        def __len__(self):
            return len(self.df.index)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.labels[idx],
            }

    def compute_metrics(eval_pred):
        def maybe(fn):
            try: return fn()
            except: return 0.0

        predictions, labels = eval_pred
        predictions = torch.tensor(predictions).squeeze().double()
        labels      = torch.tensor(labels)     .squeeze().double()

        bce = torch.nn.BCEWithLogitsLoss()(predictions, labels).item()

        predictions = torch.sigmoid(predictions) > 0.8

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

    model     = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    args = TrainingArguments(
        save_dir,
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        num_train_epochs=20,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # ratio of terrible vs all
    terrible_count = train_df["is_decent"].sum()
    not_terrible_count = len(train_df.index) - terrible_count
    pos_weight = torch.tensor(not_terrible_count / terrible_count)
    print("positive weight is", pos_weight)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits").squeeze()
            
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        model,
        args,
        train_dataset=TerribleDataset(tokenizer, train_df),
        eval_dataset=TerribleDataset(tokenizer, val_df),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(TerribleDataset(tokenizer, test_df))
    test_metrics = compute_metrics((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model

def fit_model_detect_terrible(base_model_dir, save_dir, train_df, val_df, test_df):
    def is_terrible(row):
        if row["most_common_grade"] == 0 and row["meanGrade"] <= 1:
            return 1
        elif row["most_common_grade"] == 1 and row["meanGrade"] <= 1:
            return 1
        else:
            return 0
    def is_decent(row):
        if row["meanGrade"] >= 1.5:
            return 1
        else:
            return 0
        
    dfs = [train_df, val_df, test_df]
    for i in range(len(dfs)):
        #df0 = dfs[i][dfs[i]["disagreements"] == 0]
        #df1 = dfs[i][dfs[i]["disagreements"] == 1]
        #df2 = dfs[i][dfs[i]["disagreements"] == 2]
        #dfs[i] = pd.concat([df0, df1, df2], ignore_index=True)
        dfs[i]["is_terrible"] = dfs[i].apply(is_terrible, axis=1)
        dfs[i]["is_decent"]   = dfs[i].apply(is_decent,   axis=1)

        # keep only is_terrible and is_decent rows
        dfs[i] = dfs[i][dfs[i]["is_terrible"] + dfs[i]["is_decent"] > 0]
    train_df, val_df, test_df = dfs

    class TerribleDataset(Dataset):
        def __init__(self, tokenizer, dataframe):
            self.df = dataframe
            
            original = self.df["original_sentence"].tolist()
            edited   = self.df["edited_sentence"].tolist()
            text = [f"{o} [SEP] {e}" for o, e in zip(original, edited)]
            #text = edited
            output = tokenizer(text=text, truncation=True, padding=True, return_tensors='pt')
            
            self.input_ids      = output["input_ids"]
            self.attention_mask = output["attention_mask"]
            self.labels         = torch.tensor(self.df['is_terrible'].values, dtype=torch.float32)
        def __len__(self):
            return len(self.df.index)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.labels[idx],
            }

    def compute_metrics(eval_pred):
        def maybe(fn):
            try: return fn()
            except: return 0.0

        predictions, labels = eval_pred
        predictions = torch.tensor(predictions).squeeze().double()
        labels      = torch.tensor(labels)     .squeeze().double()

        bce = torch.nn.BCEWithLogitsLoss()(predictions, labels).item()

        predictions = torch.sigmoid(predictions) > 0.85

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
        metric_for_best_model="r",
    )

    # ratio of terrible vs all
    terrible_count = train_df["is_terrible"].sum()
    not_terrible_count = len(train_df.index) - terrible_count
    pos_weight = torch.tensor(not_terrible_count / terrible_count)
    print("positive weight is", pos_weight)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits").squeeze()
            if len(logits.shape) == 0:
                logits = logits.unsqueeze(0)

            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        model,
        args,
        train_dataset=TerribleDataset(tokenizer, train_df),
        eval_dataset=TerribleDataset(tokenizer, val_df),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(TerribleDataset(tokenizer, test_df))
    test_metrics = compute_metrics((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model

#fit_model_detect_terrible('sentence-transformers/all-MiniLM-L6-v2', './fit_out', *make_base_dataset())
#fit_model_detect_decent('sentence-transformers/all-MiniLM-L6-v2', './fit_out', *make_base_dataset())

def fit_model_with_preprocessing_model_step(base_model_dir, preprocessing_model_dir, train_df, val_df, test_df):
    fit_model_detect_terrible(base_model_dir, preprocessing_model_dir, train_df, val_df, test_df)

    pp_model     = AutoModelForSequenceClassification.from_pretrained(preprocessing_model_dir)
    pp_tokenizer = AutoTokenizer.from_pretrained(preprocessing_model_dir)

    biases = np.arange(0, 2+0.1, 0.1) / 3
    #thresholds = np.arange(0.6, 1, 0.1)
    thresholds = np.array([0.85])
    for threshold in thresholds:

        def preprocess(df):
            df_kept      = df.iloc[:0,:].copy()
            df_discarded = df.iloc[:0,:].copy()
            
            for i in range(0, len(df), 1000):
                rows = df.iloc[i:i+1000]
                text = rows["edited_sentence"].tolist()
                tokenized = pp_tokenizer(text=text, truncation=True, padding=True, return_tensors='pt')
                out = pp_model(**tokenized)
                logits = out["logits"].squeeze()
                predictions = (torch.sigmoid(logits) > threshold).numpy()

                df_discarded = pd.concat([df_discarded, rows[predictions]])
                df_kept      = pd.concat([df_kept,      rows[~predictions]])

                #print("processed {} of {} rows".format(i, len(df)))

            return df_kept, df_discarded

        train_df_kept, train_df_discarded = preprocess(train_df)
        val_df_kept,   val_df_discarded   = preprocess(val_df)
        test_df_kept,  test_df_discarded  = preprocess(test_df)

        print("in train, discared {} of {} rows".format(len(train_df_discarded), len(train_df)))
        print("in val,   discared {} of {} rows".format(len(val_df_discarded),   len(val_df)))
        print("in test,  discared {} of {} rows".format(len(test_df_discarded),  len(test_df)))

        print("training on rest...")

        train_df_equally_sampled = train_df  #train_df.sample(n=len(train_df_kept), random_state=42)
        val_df_equally_sampled   = val_df    #val_df.sample(n=len(val_df_kept),     random_state=42)
        test_df_equally_sampled  = test_df   #test_df.sample(n=len(test_df_kept),   random_state=42)
        model_regular, trainer_regular = fit_regression(base_model_dir, "./second_model_out_regular", train_df_equally_sampled, val_df_equally_sampled, test_df_equally_sampled)
        trainer_regular.evaluate()

        model_kept, trainer_kept = fit_regression(base_model_dir, "./second_model_out_kept", train_df_kept, val_df_kept, test_df_kept)
        trainer_kept.evaluate()

        model_discarded, trainer_discarded = fit_regression(base_model_dir, "./second_model_out_discarded", train_df_discarded, val_df_discarded, test_df_discarded)
        #model_discarded, trainer_discarded = model_regular, trainer_regular
        trainer_discarded.evaluate()

        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

        out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df, "arrow_sentence"))

        out_kept      = trainer_kept     .predict(TokenizedSentencesDataset(tokenizer, test_df_kept,      "arrow_sentence"))
        out_discarded = trainer_discarded.predict(TokenizedSentencesDataset(tokenizer, test_df_discarded, "arrow_sentence"))

        out_combined_predictions = np.concatenate([out_kept.predictions.squeeze(), out_discarded.predictions.squeeze()])
        out_combined_label_ids   = np.concatenate([out_kept.label_ids.squeeze(),   out_discarded.label_ids.squeeze()])

        test_metrics_regular  = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
        test_metrics_combined = compute_metrics_regression((out_combined_predictions, out_combined_label_ids))
        print(f"ENTIRE dataset, regular  model  (threshold {threshold}):", test_metrics_regular)
        print(f"ENTIRE dataset, combined models (threshold {threshold}):", test_metrics_combined)

        # compare metrics like above, but for each agreement class separately
        for disagreement in [0, 1, 2, 3]:
                test_df_disagreement           = test_df[test_df["disagreements"] == disagreement]
                test_df_kept_disagreement      = test_df_kept[test_df_kept["disagreements"] == disagreement]
                test_df_discarded_disagreement = test_df[test_df["disagreements"] == disagreement]
                
                if len(test_df_disagreement) == 0:
                    continue
                
                out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df_disagreement, "arrow_sentence"))

                out_kept      = { "predictions": np.array([], dtype=float), "label_ids": np.array([], dtype=float) }
                out_discarded = { "predictions": np.array([], dtype=float), "label_ids": np.array([], dtype=float) }

                if len(test_df_kept_disagreement) != 0:
                    out_kept = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df_kept_disagreement, "arrow_sentence"))._asdict()
                if len(test_df_discarded_disagreement) != 0:
                    out_discarded = trainer_discarded.predict(TokenizedSentencesDataset(tokenizer, test_df_discarded_disagreement, "arrow_sentence"))._asdict()

                out_combined_predictions = np.concatenate([out_kept["predictions"].squeeze(), out_discarded["predictions"].squeeze()])
                out_combined_label_ids   = np.concatenate([out_kept["label_ids"].squeeze(),   out_discarded["label_ids"].squeeze()])

                test_metrics_regular  = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
                test_metrics_combined = compute_metrics_regression((out_combined_predictions, out_combined_label_ids))
                print(f"ENTIRE dataset, regular  model,  disagreement {disagreement} (threshold {threshold}):", test_metrics_regular)
                print(f"ENTIRE dataset, combined models, disagreement {disagreement} (threshold {threshold}):", test_metrics_combined)

        # like above, but for a few meanGrade buckets
        for bucket in [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]:
            mean_grade_from = bucket - 0.25
            mean_grade_to   = bucket + 0.25

            test_df_bucket           = test_df          [(test_df          ["meanGrade"] >= mean_grade_from) & (test_df          ["meanGrade"] <= mean_grade_to)]
            test_df_kept_bucket      = test_df_kept     [(test_df_kept     ["meanGrade"] >= mean_grade_from) & (test_df_kept     ["meanGrade"] <= mean_grade_to)]
            test_df_discarded_bucket = test_df_discarded[(test_df_discarded["meanGrade"] >= mean_grade_from) & (test_df_discarded["meanGrade"] <= mean_grade_to)]
            
            if len(test_df_bucket) == 0:
                continue

            out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df_bucket, "arrow_sentence"))

            out_kept      = { "predictions": np.array([], dtype=float), "label_ids": np.array([], dtype=float) }
            out_discarded = { "predictions": np.array([], dtype=float), "label_ids": np.array([], dtype=float) }
            
            if len(test_df_kept_bucket) != 0:
                out_kept = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df_kept_bucket, "arrow_sentence"))._asdict()
            if len(test_df_discarded_bucket) != 0:
                out_discarded = trainer_discarded.predict(TokenizedSentencesDataset(tokenizer, test_df_discarded_bucket, "arrow_sentence"))._asdict()

            out_combined_predictions = np.concatenate([out_kept["predictions"].squeeze(), out_discarded["predictions"].squeeze()])
            out_combined_label_ids   = np.concatenate([out_kept["label_ids"].squeeze(),   out_discarded["label_ids"].squeeze()])

            test_metrics_regular  = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
            test_metrics_combined = compute_metrics_regression((out_combined_predictions, out_combined_label_ids))
            print(f"ENTIRE dataset, regular  model,  meanGrade {mean_grade_from}-{mean_grade_to} (threshold {threshold}):", test_metrics_regular)
            print(f"ENTIRE dataset, combined models, meanGrade {mean_grade_from}-{mean_grade_to} (threshold {threshold}):", test_metrics_combined)









        
##        for bias in biases:
##            out = trainer_kept.predict(TokenizedSentencesDataset(AutoTokenizer.from_pretrained(base_model_dir), test_df_kept, "arrow_sentence"))
##
##            combined_preds  = np.concatenate((out.predictions.squeeze(), np.ones(len(test_df_discarded)) * bias))
##            combined_labels = np.concatenate((out.label_ids.squeeze(),   test_df_discarded["normalized_score"].to_numpy()))
##
##            test_metrics = compute_metrics_regression((combined_labels, combined_preds))
##            print(f"COMBINED METRICS (bias {bias}, threshold {threshold}):", test_metrics)
##            
##            out = trainer_kept.predict(TokenizedSentencesDataset(AutoTokenizer.from_pretrained(base_model_dir), test_df_kept, "arrow_sentence"))
##            test_metrics = compute_metrics_regression((out.label_ids.squeeze(), out.predictions.squeeze()))
##            print(f"KEPT METRICS (bias {bias}, threshold {threshold}):", test_metrics)
##
##            out = trainer_kept.predict(TokenizedSentencesDataset(AutoTokenizer.from_pretrained(base_model_dir), test_df_discarded, "arrow_sentence"))
##            test_metrics = compute_metrics_regression((out.label_ids.squeeze(), out.predictions.squeeze()))
##            print(f"DISCARDED METRICS (bias {bias}, threshold {threshold}):", test_metrics)
##
##            out = trainer_kept.predict(TokenizedSentencesDataset(AutoTokenizer.from_pretrained(base_model_dir), test_df, "arrow_sentence"))
##            test_metrics = compute_metrics_regression((out.label_ids.squeeze(), out.predictions.squeeze()))
##            print(f"EVALD ON FULL SET (bias {bias}, threshold {threshold}):", test_metrics)


fit_model_with_preprocessing_model_step('sentence-transformers/all-MiniLM-L6-v2', './fit_out', *make_base_dataset())


def do_full_gpl_run():
    train_df, val_df, test_df = make_base_dataset()
    train_df = load_explanations(train_df, "explanations/chatgpt_train.json", drop_without_explanation=True)
    val_df   = load_explanations(val_df,   "explanations/chatgpt_valid.json", drop_without_explanation=True)

    #gpl_dir = "gpl_workingdir_chatgpt_reversed"
    #prepare_gpl_workingdir(gpl_dir, train_df, val_df, True)
    #train_gpl(gpl_dir, "distilbert-base-uncased")

    fit_regression('sentence-transformers/all-MiniLM-L6-v2', './fit_out', train_df, val_df, test_df)


def multiple_regression_by_quantiles():
    train_df, val_df, test_df = make_base_dataset()

    class QuantilesTokenizedSentencesDataset(Dataset):
        def __init__(self, tokenizer, dataframe, quantiles=1, quantile_index=0, device="cpu"):
            self.df             = dataframe
            self.quantile_index = quantile_index

            qrange = torch.linspace(0, 1, quantiles)
            quants = torch.zeros(len(self.df), quantiles)
            for i, scores in self.df["all_scores"].items():
                quants[i] = torch.quantile(torch.tensor(scores).float(), qrange)

            original = self.df["original_sentence"].tolist()
            edited   = self.df["edited_sentence"].tolist()
            text = [f"{o} [SEP] {e}" for o, e in zip(original, edited)]
            output = tokenizer(text=text, truncation=True, padding=True, return_tensors='pt').to(device)
            
            self.input_ids      = output["input_ids"]
            self.attention_mask = output["attention_mask"]
            self.labels         = quants / 3
        def __len__(self):
            return len(self.df.index)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.labels[idx],
            }
        
    def my_metric(eval_pred):
        preds, labels = eval_pred
        preds, labels = preds.squeeze(), labels.squeeze()

        p, r, f1, _ = precision_recall_fscore_support(labels, preds.round(), average="binary")
        return {
            "cross_entropy": log_loss(labels, preds, labels=[0, 1]),
            "acc": accuracy_score(labels, preds.round()),
            "p": p,
            "r": r,
            "f1": f1,
        }


    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_fast=True)

    QUANTILES = 3
    train_dataset = QuantilesTokenizedSentencesDataset(tokenizer, train_df, quantiles=QUANTILES)
    val_dataset   = QuantilesTokenizedSentencesDataset(tokenizer, val_df,   quantiles=QUANTILES)
    test_dataset  = QuantilesTokenizedSentencesDataset(tokenizer, test_df,  quantiles=QUANTILES)

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=QUANTILES)

    args = TrainingArguments(
        f"quantile_test/q{QUANTILES}",
        save_strategy="steps",
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        num_train_epochs=10,
        warmup_ratio=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics_multi_regression(eval_pred):
        predictions, labels = eval_pred
        predictions, labels = torch.tensor(predictions).squeeze(), torch.tensor(labels).squeeze()
        predictions *= 3
        labels      *= 3
        
        predictions = predictions.mean(dim=1)
        labels      = labels.mean(dim=1)

        mse = mse_metric.compute(predictions=predictions, references=labels)
        mae = mae_metric.compute(predictions=predictions, references=labels)
        return {
            "mae":  mae["mae"],
            "mse":  mse["mse"],
            "rmse": np.sqrt(mse["mse"]),
        }

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_multi_regression,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
    )

    trainer.train()
    trainer.evaluate()
    out = trainer.predict(test_dataset)
    test_metrics = compute_metrics_multi_regression((out.predictions, out.label_ids))
    print("test_metrics", test_metrics)