
def fit_model_with_preprocessing_model_step(base_model_dir, preprocessing_model_dir, train_df, val_df, test_df):
    #fit_model_detect_terrible(base_model_dir, preprocessing_model_dir, train_df, val_df, test_df)

    pp_model     = AutoModelForSequenceClassification.from_pretrained(preprocessing_model_dir)
    pp_tokenizer = AutoTokenizer.from_pretrained(preprocessing_model_dir)

    biases = np.arange(0, 2+0.1, 0.1) / 3
    #thresholds = np.arange(0.6, 1, 0.1)
    thresholds = np.array([0.8])
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
            
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

        out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df_kept, "arrow_sentence"))
        out_kept    = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df_kept, "arrow_sentence"))
        test_metrics_regular = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
        test_metrics_kept    = compute_metrics_regression((out_kept.predictions.squeeze(),    out_kept.label_ids.squeeze()))
        print(f"KEPT   dataset, randomly sampled model (threshold {threshold}):", test_metrics_regular)
        print(f"KEPT   dataset, kept-only        model (threshold {threshold}):", test_metrics_kept)

        out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df, "arrow_sentence"))
        out_kept    = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df, "arrow_sentence"))
        test_metrics_regular = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
        test_metrics_kept    = compute_metrics_regression((out_kept.predictions.squeeze(),    out_kept.label_ids.squeeze()))
        print(f"ENTIRE dataset, randomly sampled model (threshold {threshold}):", test_metrics_regular)
        print(f"ENTIRE dataset, kept-only        model (threshold {threshold}):", test_metrics_kept)

        # compare metrics like above, but for each agreement class separately
        for disagreement in [0, 1, 2, 3]:
            try:
                test_df_kept_disagreement = test_df_kept[test_df_kept["disagreements"] == disagreement]
                test_df_disagreement      = test_df[test_df["disagreements"] == disagreement]
                if len(test_df_kept_disagreement) == 0 or len(test_df_disagreement) == 0:
                    continue

                out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df_kept_disagreement, "arrow_sentence"))
                out_kept    = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df_kept_disagreement, "arrow_sentence"))
                test_metrics_regular = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
                test_metrics_kept    = compute_metrics_regression((out_kept.predictions.squeeze(),    out_kept.label_ids.squeeze()))
                print(f"KEPT   dataset, randomly sampled model, disagreement {disagreement} (threshold {threshold}):", test_metrics_regular)
                print(f"KEPT   dataset, kept-only        model, disagreement {disagreement} (threshold {threshold}):", test_metrics_kept)

                out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df_disagreement, "arrow_sentence"))
                out_kept    = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df_disagreement, "arrow_sentence"))
                test_metrics_regular = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
                test_metrics_kept    = compute_metrics_regression((out_kept.predictions.squeeze(),    out_kept.label_ids.squeeze()))
                print(f"ENTIRE dataset, randomly sampled model, disagreement {disagreement} (threshold {threshold}):", test_metrics_regular)
                print(f"ENTIRE dataset, kept-only        model, disagreement {disagreement} (threshold {threshold}):", test_metrics_kept)
            except Exception as e:
                print(e)

        # like above, but for a few meanGrade buckets
        for bucket in [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]:
            mean_grade_from = bucket - 0.25
            mean_grade_to   = bucket + 0.25

            test_df_kept_bucket = test_df_kept[(test_df_kept["meanGrade"] >= mean_grade_from) & (test_df_kept["meanGrade"] <= mean_grade_to)]
            test_df_bucket      = test_df[(test_df["meanGrade"] >= mean_grade_from) & (test_df["meanGrade"] <= mean_grade_to)]
            if len(test_df_kept_bucket) == 0 or len(test_df_bucket) == 0:
                continue

            out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df_kept_bucket, "arrow_sentence"))
            out_kept    = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df_kept_bucket, "arrow_sentence"))
            test_metrics_regular = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
            test_metrics_kept    = compute_metrics_regression((out_kept.predictions.squeeze(),    out_kept.label_ids.squeeze()))
            print(f"KEPT   dataset, randomly sampled model, meanGrade {mean_grade_from}-{mean_grade_to} (threshold {threshold}):", test_metrics_regular)
            print(f"KEPT   dataset, kept-only        model, meanGrade {mean_grade_from}-{mean_grade_to} (threshold {threshold}):", test_metrics_kept)

            out_regular = trainer_regular.predict(TokenizedSentencesDataset(tokenizer, test_df_bucket, "arrow_sentence"))
            out_kept    = trainer_kept.predict(TokenizedSentencesDataset(tokenizer, test_df_bucket, "arrow_sentence"))
            test_metrics_regular = compute_metrics_regression((out_regular.predictions.squeeze(), out_regular.label_ids.squeeze()))
            test_metrics_kept    = compute_metrics_regression((out_kept.predictions.squeeze(),    out_kept.label_ids.squeeze()))
            print(f"ENTIRE dataset, randomly sampled model, meanGrade {mean_grade_from}-{mean_grade_to} (threshold {threshold}):", test_metrics_regular)
            print(f"ENTIRE dataset, kept-only        model, meanGrade {mean_grade_from}-{mean_grade_to} (threshold {threshold}):", test_metrics_kept)