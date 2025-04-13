import os
import sys
import json
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from utils.logger import logger


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    precision = precision_score(pred.label_ids, preds)
    recall = recall_score(pred.label_ids, preds)
    f1 = f1_score(pred.label_ids, preds)
    ret = {'precision': precision, 'recall': recall, 'f1': f1}
    return ret


def load_data(n_samples):
    data_dir = os.path.join(ROOT_DIR, 'data')
    pii_ratio = 0.5
    file_path = os.path.join(
        data_dir,
        'text_%d_%03d.json' % (n_samples, pii_ratio * 100),
    )
    if os.path.exists(file_path):
        data = json.load(open(file_path, 'r'))
        logger.info('Loaded: %s' % file_path)
    else:
        from utils.data_generator import generate_dataset
        logger.info('Genrate data, %d samples' % n_samples)
        data = generate_dataset(n_samples, pii_ratio=pii_ratio)
        with open(file_path, 'w') as fpw:
            json.dump(data, fpw)
        logger.info('Saved: %s' % file_path)
    texts = [item['value'] for item in data]
    labels = [int(item['flag']) for item in data]
    return texts, labels


def gen_dataset(texts, labels, tokenizer):
    from datasets import Dataset
    # --- Tokenize
    tokenized_inputs = tokenizer(texts,
                                 padding=True,
                                 truncation=True,
                                 return_tensors='pt')
    # --- data dict
    data_dict = {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels,
    }
    # --- dataset
    dataset = Dataset.from_dict(data_dict)
    # --- Train/Test Split
    dataset = dataset.train_test_split(test_size=0.2)
    trn_data = dataset['train']
    tst_data = dataset['test']
    return trn_data, tst_data


def gen_training_args(output_dir):
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        # evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch',
        report_to=[],
    )
    return training_args


def gen_trainer(model, training_args, trn_data, tst_data, tokenizer):
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trn_data,
        eval_dataset=tst_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer


def sys_argv():
    if len(sys.argv) != 2:
        n_samples = 100
        logger.warning('n_samples not given, n_samples = %d' % n_samples)
    else:
        n_samples = int(sys.argv[1])
    return n_samples


def res_dirs(n_samples):
    trn_name = 'pii_model_%06d' % n_samples
    logger.info('train name: %s' % trn_name)
    output_dir = os.path.join(ROOT_DIR, 'output', trn_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(ROOT_DIR, trn_name)
    return output_dir, model_path


def gen_model():
    from transformers import DistilBertTokenizerFast
    from transformers import DistilBertForSequenceClassification
    mdl_name = 'distilbert-base-uncased'
    # --- Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(mdl_name)
    # --- Model
    model = DistilBertForSequenceClassification.from_pretrained(mdl_name)
    return model, tokenizer


def main():
    logger.setLevel('INFO')
    n_samples = sys_argv()
    output_dir, model_path = res_dirs(n_samples)
    # --- data
    texts, labels = load_data(n_samples)
    # --- model
    model, tokenizer = gen_model()
    # --- dataset
    trn_data, tst_data = gen_dataset(texts, labels, tokenizer)
    # --- Training arguments
    train_args = gen_training_args(output_dir)
    # --- Trainer
    trainer = gen_trainer(model, train_args, trn_data, tst_data, tokenizer)
    # --- train
    trainer.train()
    # --- save model
    trainer.save_model(model_path)
    # --- Evaluate final performance
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == '__main__':
    main()
