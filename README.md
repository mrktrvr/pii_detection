# Human Native Take-Home Assignment

## Overview
This project provides modules that automatically flag text data potentially containing Personally Identifiable Information (PII) using Python. It includes rule-based, transformer-based, and supervised learning approaches.

## Assumptions

- **Data type**: Only textual data is handled in this implementation.

- **Detection approaches**: Both rule-based and ML-based techniques are implemented.

- **Synthetic data**: Generated using the `faker` library and optionally GPT-2 to simulate realistic articles.

- **PII types considered**: Name, email, address, and phone number.

- **Labelling**: Each sample is labelled with a boolean `flag` indicating whether it contains PII.

- **Evaluation**: Precision, recall, and F1 score calculated using `scikit-learn`.

- **Scalability and persistence**: Not addressed (per assignment scope).


## Detection methods

Three approaches are implemented:

### 1. Rule-Based Detection

- Uses regular expressions to detect patterns resembling PII.

### 2. Transformer-Based Pretrained Model

- Uses Hugging Face's `pipeline("ner", aggregation_strategy="simple")` with the `dslim/bert-base-NER` model.
- Performs Named Entity Recognition (NER), returning entity types such as `PER` (person), `LOC` (location), and `ORG` (organization) that may indicate PII.

### 3. Trained Classifier Model

- A `DistilBERT` based sequence classifier fine-tuned on synthetic data labelled for PII presence.
- Learns contextual patterns in text that suggest the presence of PII.
- Produces a binary prediction for each article: contains PII or not.

## Structure

```bash
.
├── main.py                        # Runs all PII detection approaches and evaluates them
├── models/
│   ├── rule_based.py              # Regex-based PII detection
│   ├── trained_model_detector.py  # PII detector using fine-tuned DistilBERT
│   ├── train_model.py             # Trains DistilBERT on generated data
│   └── transformer_model.py       # PII detection using pretrained NER model
├── utils/
│   ├── data_generator.py          # Generates synthetic text data
│   └── logger.py                  # Custom logging module
├── tests/
│   ├── test_all.bash              # Bash script to run all tests
│   ├── test_data_generator.py     # Unit tests for synthetic data generation
│   └── test_pii_detectors.py      # Unit tests for all detection models
├── requirements.txt
└── README.md
```



- `utils.data_generator.py`: Generates synthetic articles with/without PII.
- `models.rule_based.py`: Uses regular expressions to detect PII.
- `models.transformer_model.py`: 
- `main.py`: 

### Unit test

- `tests.test_data_generator.py`: Unit tests for synthetic data generation.
- `tests.test_pii_detectors.py`: 

## How to Run
### 1. Setup python Environment

```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

### 2. Install dependencies

```bash
$ pip install -r requirements.txt
```
### 3. Train the Model
```bash
$ python models/train_model.py n_samples
```
- Generates n_samples of synthetic data.
- Trains and saves a fine-tuned `DistilBERT` model to ./pii_model_[n_samples].

### Run the model and evaluation

Runs all 3 detection models and prints evaluation metrics.

```bash
$ python main.py
```
### Run tests

```bash
$ python tests/test_data_generator.py
$ python tests/test_pii_detectors.py
# or
$ bash tests/test_all.bash
```

## Evaluation
Evaluation is performed in `main.py` using a synthetic test set. Metrics used: **Precision**, **Recall**, and **F1 Score**.

| Model Name              | Precision | Recall | F1     |
| ----------------------- | --------- | ------ | ------ |
| Rule-Based Model        | 0.3000    | 1.0000 | 0.4615 |
| Transformer-Based Model | 0.7727    | 0.5667 | 0.6538 |
| Trained Model Detector  | 1.0000    | 0.9000 | 0.9474 |

The trained model achieved the highest overall performance, but may be overfitting due to limited variation in the synthetic dataset.

## Future Work

- Improve data generation with more diverse sentence structures and edge cases.
- Fine-tune transformer-based NER models for higher precision and recall.
- Expand the solution to handle other data modalities (images, audio, video).
- Explore semi-supervised or human-in-the-loop approaches using data that were verified as correct by the operations team
- Real-world data training and testing
