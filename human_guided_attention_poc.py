"""Human-guided attention in Transformers."""
from typing import Dict

import torch
from datasets import Dataset
# load imdb
from datasets import load_dataset
from essential_generators import DocumentGenerator
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, \
    matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb


def gen_synthetic_data(n: int = 1_000, p: float = 0.5, max_seq_length: int = 200, mode: str = 'normal'):
    """Synthetic data, random text, but I add in a sentence that says "yes" or "no" (lots of possible formulations, NLP increase).
    :param n: number of samples
    :param p: probability of positive sentence
    :param mode: 'normal' or 'baseline' or 'theorical_max'
    :return: list of strings
    """
    # Generate random text
    gen = DocumentGenerator()
    if mode == 'normal' or mode == 'baseline':
        texts = [gen.paragraph(min_sentences=5, max_sentences=10).split()[:max_seq_length] for _ in range(n)]
    else:  # mode == 'theorical_max':
        texts = [[] for _ in range(n)]
    # todo : try to generate abstract with more scientific terms

    # Seed sentences must be diverse
    positive_sentences = [
        "we successfully replicated",
        "we were able to replicate",
        "we were able to reproduce",
        "we were able to reproduce the results",
        "This study replicated the previous finding",
        "The authors replicated previously reported findings",
        "The authors were able to replicate the previous finding",
        "This experiment replicated the past results",
        "previous findings were replicated",
        "the previous results were replicated",
        "we managed to replicate the claims",
        "we had success replicating the findings of",
    ]

    negative_sentences = [
        "we were unable to replicate",
        "we were unable to replicate the results",
        "failure to replicate",
        "unsuccessful attempts to replicate",
        "we were unable to reproduce the results",
        "failing to replicate the main findings",
        "This study failed to replicate the previous finding",
        "Our experiment failed to replicate the past results",
        "Our results does not replicate the previous findings",
        "we found no evidence to support the claims",
        "we found that previous findings could not be replicated",
    ]

    # Add in the positive or negative sentences, at any place in the normal text
    import random
    labels = []
    masks = []
    for i in range(n):
        if random.random() < p:
            insert_idx = random.randint(0, len(texts[i]))
            insert_sentence = random.choice(positive_sentences).split()
            texts[i] = texts[i][:insert_idx] + insert_sentence + texts[i][insert_idx:]
            labels.append(1)
            masks.append((insert_idx, insert_idx + len(insert_sentence)))
        else:
            texts[i].insert(random.randint(0, len(texts[i])), random.choice(negative_sentences))
            labels.append(0)
            masks.append(None)

    # Convert to strings
    texts = [" ".join(text) for text in texts]

    return texts, labels, masks


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_train_samples', type=int, default=10_000)
    parser.add_argument('--mode', type=str, default='baseline')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_length', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)

    # to dict
    config = parser.parse_args()
    config: Dict = vars(config)
    # }
    texts, labels, masks = gen_synthetic_data(n=config['nb_train_samples'], mode=config['mode'],
                                              max_seq_length=config['max_seq_length'])

    # Use ALBERT-large for Text Classification
    model_name = "albert-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # dataset from texts
    data = tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=False)  # Truncation : false BECAUSE what if we remove the only useful part for classif!?
    data['label'] = torch.tensor(labels)
    print('length', len(data.encodings[0].word_ids))

    # construct the tokenized_masks from masks and inputs.encodings[i].word_ids
    tokenized_masks = []
    for i in range(len(masks)):
        if masks[i] is None:
            # Only zeros
            tokenized_masks.append(torch.zeros(len(data.encodings[i].word_ids)))
        else:
            # 1s where the mask is, 0s elsewhere
            begin, end = masks[i]
            tokenized_masks.append(
                torch.tensor([1 if j in range(begin, end) else 0 for j in range(len(data.encodings[i].word_ids))]))

    # Add the tokenized_masks to the dataset
    data['tokenized_masks'] = torch.stack(tokenized_masks)

    # Make a huggingface dataset from the data, and split it.
    dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)

    # train the model
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=config['epochs'],  # total number of training epochs
        per_device_train_batch_size=config['batch_size'],  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        learning_rate=config['lr'],
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,  # log every 10 steps
        evaluation_strategy="steps",
        eval_steps=10,  # evaluate every 10 steps
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        report_to="wandb",
        greater_is_better=False,
        # run_name="human-guided-attention",
    )

    with wandb.init(project='human_guided_attention', entity="alexandrerfst", group=config['mode'], config=config,
                    job_type='train', tags=[config['mode']], name=f"{config['mode']}{config['nb_train_samples']}"):  # notes=...
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=dataset["train"],  # training dataset
            eval_dataset=dataset["test"],  # evaluation dataset
            # callbacks=[WandbCallback()],  # redundant with report_to="wandb"
            # as much metrics as possible
            compute_metrics=lambda p: {
                "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean(),
                "f1": f1_score(p.label_ids, p.predictions.argmax(-1)),
                "precision": precision_score(p.label_ids, p.predictions.argmax(-1)),
                "recall": recall_score(p.label_ids, p.predictions.argmax(-1)),
                'roc_auc': roc_auc_score(p.label_ids, p.predictions.argmax(-1)),
                'cohen_kappa': cohen_kappa_score(p.label_ids, p.predictions.argmax(-1)),
                'matthews_corrcoef': matthews_corrcoef(p.label_ids, p.predictions.argmax(-1)),
                # 'confusion_matrix': confusion_matrix(p.label_ids, p.predictions.argmax(-1)),  # 4 coeffs : TN, FP, FN, TP
            }
        )
        trainer.train()

        # evaluate the model
        # trainer.evaluate()
