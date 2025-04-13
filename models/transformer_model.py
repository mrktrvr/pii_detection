import os
import sys
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


class TransformerPIIDetector:

    def __init__(self):
        self.pipeline = self._ner_pipeline()
        # self.pipeline = self._bert_base_ner_pipeline()

    def _ner_pipeline(self):
        mdl_name = 'dslim/bert-base-NER'
        pl = pipeline('ner', model=mdl_name, aggregation_strategy='simple')
        return pl

    def _bert_base_ner_pipeline(self):
        mdl_name = 'dslim/bert-base-NER'
        tokenizer = AutoTokenizer.from_pretrained(mdl_name)
        model = AutoModelForTokenClassification.from_pretrained(mdl_name)
        pl = pipeline(
            'ner',
            model=model,
            tokenizer=tokenizer,
            grouped_entities=True,
        )
        return pl

    def detect(self, text):
        entities = self.pipeline(text)
        for entity in entities:
            if entity['entity_group'] in ['PER', 'LOC', 'ORG']:
                return True
        return False


def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, ROOT_DIR)
    from utils.data_generator import generate_dataset
    n_samples = 10
    data = generate_dataset(n_samples, pii_ratio=0.3)
    # --- transformer based model
    detector = TransformerPIIDetector()
    for data in data:
        res = detector.detect(data['value'])
        print('\n'.join([
            '-' * 10,
            'Result: %5s' % res,
            'Answer: %5s' % data['flag'],
            '- text -',
            data['value'],
            '-' * 3,
            '-' * 10,
        ]))


if __name__ == '__main__':
    main()
