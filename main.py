from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils.data_generator import generate_dataset
from utils.logger import logger
from models.rule_based import detect_pii as detect_pii_rule_based
from models.transformer_model import TransformerPIIDetector
from models.trained_model_detector import TrainedModelDetector


def evaluate_model(data, detector):
    y_true = [item['flag'] for item in data]
    y_pred = [detector(item['value']) for item in data]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1


def main():
    logger.setLevel('INFO')
    results = {}
    # --- data
    n_samples = 100
    data = generate_dataset(n_samples, pii_ratio=0.3)
    # --- rule based approach
    logger.info('Rule based')
    precision, recall, f1 = evaluate_model(data, detect_pii_rule_based)
    results['Rule-Based Model'] = (precision, recall, f1)
    # --- transformer based model
    logger.info('Transformer based')
    detector = TransformerPIIDetector()
    precision, recall, f1 = evaluate_model(data, detector.detect)
    results['Transformer-Based Model'] = (precision, recall, f1)
    # --- trained model detector
    logger.info('Trained model')
    trained_detector = TrainedModelDetector()
    precision, recall, f1 = evaluate_model(data, trained_detector.detect)
    results['Trained Model Detector'] = (precision, recall, f1)
    # --- show results
    print_text_list = ['']
    print_text_list.append('|'.join([
        '%25s' % 'Model Name',
        '%6s' % 'Prec',
        '%6s' % 'Recall',
        '%6s' % 'F1',
    ]))
    print_text_list.append('|'.join(['-' * 25, '-' * 6, '-' * 6, '-' * 6]))
    for k, v in results.items():
        text = '|'.join(['%25s' % k] + ['%6.4f' % vv for vv in v])
        print_text_list.append(text)
    print_text_list = ['|%s|' % x for x in print_text_list]
    print_text_list = ['-' * 100] + print_text_list
    print_text_list = print_text_list + ['-' * 100]
    print('\n'.join(print_text_list))


if __name__ == '__main__':
    main()
