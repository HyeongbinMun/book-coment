import os
import pandas as pd
import argparse

def csvtotext(args):
    data_path = args.data_path
    data_type = args.data_type

    file_list = os.listdir(data_path)
    for file in file_list:
        file_name, _ = file.split('.')
        file_path = os.path.join(data_path, file)
        data_list = pd.read_excel(file_path)

        save_path = data_path + file_name + '.txt'
        f = open(save_path, 'w')

        for i in range(len(data_list['Content'])):
            if data_type == 'train':
                subject = 'id' + '\t' + 'document' + '\t' + 'label' + '\n'
                f.write(subject)

                if data_list['label'][i] == 'N': label = '0'
                elif data_list['label'][i] == 'P': label = '1'
                else: label = '2'
                content = data_list['Content'][i].replace('\n', ' ')
                data = str(data_list['ID'][i]) + '\t' + content + '\t' + label + '\n'
                f.write(data)
            else:
                subject = 'id' + '\t' + 'document' + '\n'
                f.write(subject)

                content = data_list['Content'][i].replace('\n', ' ')
                data = str(data_list['ID'][i]) + '\t' + content + '\t' + '\n'
                f.write(data)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/workspace/data/train.xlsx')
    parser.add_argument('--data_type', type=str, default='test', help='train or test')
    args = parser.parse_args()

    csvtotext(args)