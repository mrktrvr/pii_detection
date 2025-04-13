import os
import sys
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from utils.logger import logger


def gen_data_core(n_samples, pii_ratio, file_path):
    from utils.data_generator import generate_dataset
    data = generate_dataset(n_samples, pii_ratio=pii_ratio)
    with open(file_path, 'w') as fpw:
        json.dump(data, fpw)
        logger.info('Saved: %s' % file_path)


def main():
    logger.setLevel('INFO')
    data_dir = os.path.join(ROOT_DIR, 'data')
    params = [
        (200, 0.3),
        (200, 0.5),
        (100, 0.5),
    ]
    for n_samples, pii_ratio in params:
        file_path = os.path.join(
            data_dir,
            'text_%03d_%03d.json' % (n_samples, pii_ratio * 100),
        )
        if os.path.exists(file_path):
            logger.warning('%s exists. Skip generating data.' % file_path)
        else:
            logger.info('Creating %s.' % file_path)
            gen_data_core(n_samples, pii_ratio, file_path)


if __name__ == '__main__':
    main()
