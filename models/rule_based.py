import os
import sys
import re


def detect_pii(text):
    r'''
    Regular expression logics
    - email
    [\w.-]+: matches one or more alphanumeric characters (a-z, A-Z, 0-9),
    underscores, dots, or hyphens.
    \.\w+: matches a period followed by the top-level domain (e.g. .com, .org).
    - phone
    \b: word boundary, to make sure it’s a standalone number.
    \d{10,15}: matches a string of 10 to 15 digits.
    \b: another word boundary.
    Note that this expression doesn't support international number like +44.
    - address
    \d{1,5}: matches a number with 1~5 digits (e.g. house number).
    \s: matches a space.
    \w+\s\w+: matches two consecutive words (e.g. street name).
    - name
    \b: word boundary.
    (Mr\.|Mrs\.|Ms\.|Dr\.)?: optional title.
    \s?: optional space after the title.
    [A-Z][a-z]+: capitalised word (e.g., a name like Smith).
    \b: word boundary.
    Note that this expression maches capitalised wors like London or Word.
    '''
    patterns = {
        'email': r'[\w.-]+@[\w.-]+\.\w+',
        'phone': r'\b\d{10,15}\b',
        'address': r'\d{1,5}\s\w+\s\w+',
        'name': r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)?\s?[A-Z][a-z]+\b',
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text):
            return True
    return False


def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, ROOT_DIR)
    from utils.data_generator import generate_dataset
    n_samples = 10
    data = generate_dataset(n_samples, pii_ratio=0.3)
    # --- transformer based model
    for data in data:
        res = detect_pii(data['value'])
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
