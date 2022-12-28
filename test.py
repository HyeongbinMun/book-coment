import numpy as np
import torch
import argparse

from train import BERTDataset
from train import BERTClassifier
from kobert import get_pytorch_kobert_model

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/workspace/result/last.pt')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()


    end = 1
    while end == 1:
        sentence = input("하고싶은 말을 입력해주세요 : ")
        if sentence == "0":
            break
        predict(sentence)
        print("\n")

    predict()