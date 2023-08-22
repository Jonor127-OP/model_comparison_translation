import argparse
import gzip
import numpy as np

def load_data(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]

def calculate_metrics(data):
    sequence_lengths = [len(sequence) for sequence in data]
    mean_length = np.mean(sequence_lengths)
    std_length = np.std(sequence_lengths)
    variance = np.var(sequence_lengths)
    
    return {
        'mean_length': mean_length,
        'std_length': std_length,
        'variance': variance,
        'num_sequences': len(data),
        'total_tokens': sum(sequence_lengths)
    }

def main(args):
    if args.dataset == 1:
        data_file_path = 'dataset/nl/lm/en2de/wmt17_en_de/train.merge_en_de.ids.gz'
    elif args.dataset == 2:
        data_file_path = 'dataset/nl/lm/en2fr/wmt14_en_fr/train.merge_en_fr.ids.gz'
    else:
        raise ValueError("Invalid dataset option. Choose 1 or 2.")
    
    data = load_data(data_file_path)
    metrics = calculate_metrics(data)
    
    print("Nombre de séquences:", metrics['num_sequences'])
    print("Nombre total de tokens:", metrics['total_tokens'])
    print("Longueur moyenne de séquence:", metrics['mean_length'])
    print("Écart type de la longueur de séquence:", metrics['std_length'])
    print("Variance de la longueur de séquence:", metrics['variance'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics for dataset')
    parser.add_argument('--dataset', type=int, choices=[1, 2], default=1,
                        help='Dataset option: 1 for wmt17_en_de, 2 for wmt14_en_fr')
    args = parser.parse_args()
    main(args)
