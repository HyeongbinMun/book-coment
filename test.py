import torch
import argparse
import gluonnlp as nlp
import numpy as np

from kobert import get_tokenizer
from train import BERTDataset
from train import BERTClassifier
from kobert import get_pytorch_kobert_model

def file_read(file_path):
    data_list = []
    count = 0
    f = open(file_path, 'r')
    while True:
        line = f.readline()
        if not line: break
        if count != 0:
            new_str = line.replace("\n", "")
            id, content, label = new_str.split('\t')
            data = {'id': id, 'content': content, 'label': label}
            data_list.append(data)
        count += 1
    f.close()

    return data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_path', type=str, default='/workspace/result/ver2/last.pt')
    parser.add_argument('--test_path', type=str, default='/workspace/data/test.txt')
    parser.add_argument('--test_type', type=str, default='file', help='file or str')
    parser.add_argument('--test_str', type=str, default='이병헌목소리대박이네', help='file or string')
    parser.add_argument('--save_path', type=str, default='/workspace/result/')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_data = []
    if args.test_type == 'file':
        test_list = file_read(args.test_path)
        for el in test_list:
            tmp = []
            tmp.append(el['content'])
            tmp.append(el['label'])
            test_data.append(tmp)
    else:
        test_list = [args.test_str, '0']
        test_data.append(test_list)

    save_file = args.save_path + 'result.txt'
    f = open(save_file, 'w', encoding="UTF-8")
    subject = 'id' + '\t' + 'gt' + '\t' + 'predict' + '\n'
    f.write(subject)

    for element in test_data:
        data_list = [element]
        test = BERTDataset(data_list, 0, 1, tok, args.max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=5, shuffle=False)

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)
            logits = out
            logits = logits.detach().cpu().numpy()
            result = element[0] + '\t' + element[1] + '\t' + str(np.argmax(logits)) + '\n'
            print(result)
            f.write(result)

    f.close()
