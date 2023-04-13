# tar-project

## Resources
- [Humicroset Paper](https://arxiv.org/pdf/1906.00274v1.pdf)
- [Humicroset Dataset](https://huggingface.co/datasets/humicroedit)

## Random notes & commentary

What I tried so far:
- Use precomputed "static" sentence embeddings, append a linear layer as the classification head, train just the linear layer. I tried modelling it as a regression task, which gives me a bad vibe. We lose a lot of information this way. Is the joke polarizing? Does some dumbo just find it funny and noone else? Or is the joke truly funny/unfunny?
- Tried both MSE and MAE loss. MSE loss doesn't perform good for small variations, MAE doesn't perform good for things which a model should be able to strongly assert about. Write this up in more detail.
- Did the same thing, but tried to model the probability distribution of all scores. Here we can use CrossEntropyLoss and sleep better at night. Smoothing the labels didn't seem to help.
- Tried all of the above, but instead of using pooled embeddings for each token, used just the embeddings for the edited word (did a reverse lookup substring->token indices). This performed marginally better, but wouldn't say that it's statistically significant.
- Tried finetuning distilbert instead, this performs somewhat better. We still need a better way of evaluating this, I don't really understand what they do in the paper.
    
Question for the checkpoint:
- **How should we evaluate our models?** I don't really understand the bucketing approach used in the paper.
- **What is the best way to model scoring?** While at first this seems like a regression task, I don't really think it is. Sure, we have a real number, but our number is tightly bound between 0 and 1 - MSE/regression can't make use of this information. The real number is also not an "actual" real number, it comes from averaging discreete values. Is there a better way to do this?
- Logistic regression also doesn't feel correct here (just a hunch) - we derived it for a binary classification task, it isn't clear that this transfers to a score that looks like a probability.
- Modeling the probability distribution of all scores seems like a better approach in theory. Given that scores are largely subjective it does make some sense to think about a "probability that someone will label this as very funny". People are also using MulticlassCrossEntropyLoss with probabilities, instead of clear labels, so this also seems fine.
- However, here we run into the issue of our dataset - the jokes really aren't funny, and the annotator exhaustion really comes through in the labels. The dataset is very "lukewarm" - most samples fall into the "barely funny" category, and some similar jokes are rated quite differently. There is a ton of annotators, but each sample is rated by just a few. This makes it very hard to confidently model annotators and their sense of humor. **Could we somehow model this? Are annotators usually modeled, and if so, how? Could we somehow apply some apriori knowledge to make our uncertain label probabilities more assertive?**
