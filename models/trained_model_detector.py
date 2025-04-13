import os
import sys
from glob import glob
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from utils.logger import logger


class TrainedModelDetector:

    def __init__(self, model_path=''):
        if model_path == '':
            mdl_paths = glob(os.path.join(ROOT_DIR, 'pii_model_*'))
            if len(mdl_paths) == 0:
                logger.error('Model not ound. pii_model_######')
                sys.exit()
            else:
                model_path = sorted(mdl_paths)[-1]
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path)
        self.model.eval()

    def detect(self, text):
        '''
        Returns True if PII is detected.
        '''
        inputs = self.tokenizer(text,
                                return_tensors='pt',
                                truncation=True,
                                padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs).item()
        # --- 1 = PII, 0 = non-PII
        res = prediction == 1
        return res


def main():
    text = 'Please contact me at m.e@example.com.'
    trained_detector = TrainedModelDetector()
    result = trained_detector.detect(text)
    print('%s -> %s' % (text, result))


if __name__ == '__main__':
    main()
