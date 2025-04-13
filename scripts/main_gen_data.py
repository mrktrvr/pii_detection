from utils.data_generator import generate_dataset


def main():
    # --- data
    n_samples = 200
    pii_ratio = 0.3
    data = generate_dataset(n_samples, pii_ratio=pii_ratio)


if __name__ == '__main__':
    main()
