import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

from tqdm import tqdm
import argparse
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=3,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_path', type=str, default='/workspace/data/train.txt')
    parser.add_argument('--val_path', type=str, default='/workspace/data/val.txt')
    parser.add_argument('-save_path', '--save_path', type=str, default='/workspace/result/')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

    dataset_train = nlp.data.TSVDataset(args.train_path, field_indices=[1, 2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset(args.val_path, field_indices=[1, 2], num_discard_samples=1)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data_train = BERTDataset(dataset_train, 0, 1, tok, args.max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, args.max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, num_workers=5)

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * args.num_epochs
    warmup_step = int(t_total * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    for e in range(args.num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        best_te_loss = 100.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % args.log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

        if loss < best_te_loss:
            best_te_loss = loss
            best_ep = e
            torch.save(model.state_dict(), args.save_path + "best.pt")
        torch.save(model.state_dict(), args.save_path + "last.pt")