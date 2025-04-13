# Personally Identifiable Information Detection

## Overview
This project provides modules that automatically flag text data potentially containing Personally Identifiable Information (PII) using Python. It includes rule-based, transformer-based, and supervised learning approaches.

## Assumptions

- **Data type**: Only textual data is handled in this implementation.

- **Detection approaches**: Both rule-based and ML-based techniques are implemented.

- **Synthetic data**: Generated using the `faker` library and optionally `GPT-2` to simulate realistic articles.

- **PII types considered**: Name, email, address, and phone number.

- **Labelling**: Each sample is labelled with a Boolean `flag` indicating whether it contains PII.

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
├── models/
│   ├── rule_based.py              # Regular expression based PII detection
│   ├── trained_model_detector.py  # PII detector using fine-tuned DistilBERT
│   └── transformer_model.py       # PII detection using pretrained NER model
├── utils/
│   ├── data_generator.py          # Generates synthetic text data with / without PII.
│   └── logger.py                  # Custom logging module
├── scripts
│   ├── main_gen_data.py           # Generate data and store in json file
│   ├── main.py                    # Runs all PII detection approaches and evaluates them
│   └── train_model.py             # Trains DistilBERT on generated data
├── tests/
│   ├── test_all.bash              # Bash script to run all tests
│   ├── test_data_generator.py     # Unit tests for synthetic data generation
│   └── test_pii_detectors.py      # Unit tests for all detection models
├── data
│   ├── text_200_030.json          # Data which is used in main.py
│   └── text_200_050.json          # Data which is used in train_model.py
├── requirements.txt
└── README.md
```

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
### 3. Run tests

```bash
$ python tests/test_data_generator.py
$ python tests/test_pii_detectors.py
# or
$ bash tests/test_all.bash
```

### 4. Train the Model

```bash
$ python scripts/train_model.py n_samples
```

- Generates n_samples of synthetic data.
- Trains and saves a fine-tuned `DistilBERT` model to ./pii_model_[n_samples].

### 5. Run the model and evaluation

Runs all 3 detection models and prints evaluation metrics.

```bash
$ python scripts/main.py
```

## Evaluation

Evaluation is performed in `main.py` using a synthetic test set. Metrics used: **Precision**, **Recall**, and **F1 Score**.

| Model Name              | Prec   | Recall | F1     |
| ----------------------- | ------ | ------ | ------ |
| Rule-Based Model        | 0.2700 | 1.0000 | 0.4252 |
| Transformer-Based Model | 0.7714 | 1.0000 | 0.8710 |
| Trained Model Detector  | 1.0000 | 0.8519 | 0.9200 |

The trained model achieved the highest overall performance, but may be over-fitting due to limited variation in the synthetic texts in both training data and test data.

## Future Work

- Improve data generation with more diverse sentence structures and edge cases.
- Fine-tune transformer-based NER models for higher precision and recall.
- Expand the solution to handle other data modalities (images, audio, video).
- Explore semi-supervised or human-in-the-loop approaches using data that were verified as correct by the operations team
- Real-world data training and testing
