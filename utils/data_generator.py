import os
import sys
import random
import uuid
from faker import Faker
from transformers import pipeline

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from utils.logger import logger

fake = Faker()
Faker.seed(0)


def generate_pii():
    if True:
        pii_fields = ['name', 'company', 'location']
        pii_type = random.choice(pii_fields)
        if pii_type in ['name', 'company']:
            ret = getattr(fake, pii_type)()
        elif pii_type == 'location':
            ret = 'Location: %s, %s' % (fake.city(), fake.country())
        else:
            ret = 'PII'
        return ret
    else:
        pii_fields = ['name', 'address', 'email', 'phone_number']
        pii_type = random.choice(pii_fields)
        ret = getattr(fake, pii_type)()
        return ret


text_generator = pipeline('text-generation', model='gpt2')


def generate_sample(value_contains_pii=True):
    src_txt_list = [
        'Better AI starts with better data.',
        'We bring high quality AI.',
        'Premium data with reputable AI developers—come join the ecosystem.',
        'We are creating a fair data ecosystem by AI.',
        'The AI ecosystem ensures equitable compensation for rights holders.',
        'While enabling responsible AI development',
    ]
    if text_generator is None:
        text = random.choice(src_txt_list)
    else:
        prompt = random.choice(src_txt_list)
        text_gen = text_generator(
            prompt,
            max_length=20,
            do_sample=True,
            pad_token_id=50256,
            truncation=True,
        )
        text = text_gen[0]['generated_text']
    if value_contains_pii:
        if 'AI' in text:
            text = text.replace('AI', generate_pii())
        else:
            half = len(text) // 2
            text = ' '.join([text[:half], generate_pii(), text[half:]])
    return text


def generate_dataset(num_samples=1000, pii_ratio=0.3):
    '''
    dataset: list of dictionary
            dataset_id: dataset_id
            id: data_id
            value: text data,
            flag: bool, has pii or not
    '''
    logger.info('Gerate %d samples(pii_ratio=%.2f)' % (num_samples, pii_ratio))
    dataset = []
    for _ in range(num_samples):
        has_pii = random.random() < pii_ratio
        entry = {
            'dataset_id': str(uuid.uuid4()),
            'id': str(uuid.uuid4()),
            'value': generate_sample(has_pii),
            'flag': has_pii
        }
        dataset.append(entry)
    logger.info('Generated %d samples' % num_samples)
    return dataset


def main():
    n_texts = 10
    pii_ratio = 0.5
    data = generate_dataset(n_texts, pii_ratio)
    for i, d in enumerate(data):
        print('--- text %03d, is PII %5s ---' % (i, d['flag']))
        print('```')
        print('%s' % d['value'])
        print('```')
        print('-' * 100)


if __name__ == '__main__':
    main()
