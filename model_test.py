import logging
import numpy as np
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
        eval_steps=500,
        num_train_epochs=10,
        warmup_ratio=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
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
    return model


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

def do_full_gpl_run():
    train_df, val_df, test_df = make_base_dataset()
    train_df = load_explanations(train_df, "explanations/chatgpt_train.json", drop_without_explanation=True)
    val_df   = load_explanations(val_df,   "explanations/chatgpt_valid.json", drop_without_explanation=True)

    gpl_dir = "gpl_workingdir_chatgpt_reversed"
    #prepare_gpl_workingdir(gpl_dir, train_df, val_df, True)
    #train_gpl(gpl_dir, "distilbert-base-uncased")

    fit_model_annotators('bert-base-uncased', './fit_out', train_df, val_df, test_df)


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