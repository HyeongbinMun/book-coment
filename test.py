import numpy as np
import torch
import argparse
import gluonnlp as nlp

from kobert import get_tokenizer
from train import BERTDataset
from train import BERTClassifier
from kobert import get_pytorch_kobert_model

def file_read(file_path):
    data_list = []
    f = open(file_path, 'r')
    while True:
        line = f.readline()
        if not line: break
        new_str = line.replace("\n", "")
        id, content, label = new_str.split('\t')
        data = {'id': id, 'content': content, 'label': label}
        data_list.append(data)
    f.close()

    return data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='/workspace/result/last.pt')
    parser.add_argument('--test_path', type=str, default='/workspace/data/test.txt')
    parser.add_argument('--save_path', type=str, default='/workspace/result/')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_list = file_read(args.test_path)
    test_data = []
    for el in test_list:
        data = []
        data.append(el['content'])
        data.append(el['label'])
        test_data.append(data)

    another_test = BERTDataset(test_data, 0, 1, tok, vocab, args.max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=args.batch_size, num_workers=5)

    save_file = args.save_path + 'result.txt'
    f = open(save_file, 'w')

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        logits = out
        logits = logits.detach().cpu().numpy()
        result = segment_ids + '\t' + logits + '\n'
        f.write(result)

    f.close()
