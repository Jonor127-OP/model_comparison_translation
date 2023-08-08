import csv
import gzip
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import TextSamplerDatasetLM, MyCollateLM, ids_to_tokens, BPE_to_eval, remove_eos, get_input_output_lm
from model.lm import LanguageModel
import re

def compare_sources_targets(model_path, dataset_path, vocabulary_path, max_len, window_training, output_file):
    with open(vocabulary_path, 'r') as f:
        vocabulary = json.load(f)
    reverse_vocab = {id: token for token, id in vocabulary.items()}
    num_tokens = len(reverse_vocab.keys())

    # Load the model
    model = LanguageModel(
        model_dimension=512,
        vocab_size=num_tokens,
        number_of_heads=4,
        number_of_layers=2,
        dropout_probability=0.1
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the dataset
    with gzip.open(dataset_path, 'rb') as file:
        data = file.read()
        data = data.decode(encoding='utf-8')
        data = data.split('\n')
        data = [np.array([int(x) for x in line.split()]) for line in data if line != '']
        dataset = TextSamplerDatasetLM(data, max_len)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=MyCollateLM(pad_idx=0))

    pairs = []
    for i, batch in enumerate(data_loader):
        source, target = get_input_output_lm(batch, window=window_training)
        source_tokens = ids_to_tokens(source.tolist()[0][1:], vocabulary)
        target_tokens = ids_to_tokens(target.tolist()[0][1:], vocabulary)

        # Remove special characters using regex
        source_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(source_tokens))
        target_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(target_tokens))

        pairs.append([source_str, target_str])

    # Export the results to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['No', 'Source', 'Target'])  # Write the header rown    

        for i, pair in enumerate(pairs):
            writer.writerow([i+1, pair[0], pair[1]])

    print('Comparison results exported to', output_file)


if __name__ == '__main__':
    model_path = 'output/model_lm.pt'
    dataset_path = 'dataset/nl/lm/wmt17_en_de/valid.merge_en_de.ids.gz'
    vocabulary_path = 'dataset/nl/lm/wmt17_en_de/vocabulary.json'
    max_len = 200
    window_training = 0
    output_file = 'comparison_results.csv'

    compare_sources_targets(model_path, dataset_path, vocabulary_path, max_len, window_training, output_file)

    print('Comparison results exported to', output_file)
