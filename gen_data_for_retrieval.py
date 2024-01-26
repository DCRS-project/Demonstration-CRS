import pandas as pd
import json
import pickle
from tqdm import tqdm
from transformers import BertTokenizer


def read_pickle_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def read_json_file(path):
    with open(path, 'r') as f:
        data = f.readlines()
        return data

def create_retrieval_training_data(json_file_path, out_path):
    
    data = read_json_file(json_file_path)
    data = pre_process(data)
    new_data = {
        "sent0": [],
        "sent1": []
    }
    for line in data:
        new_data["sent0"].append(line['context'])
        new_data['sent1'].append(line['resp'])
    
    df = pd.DataFrame(new_data)
    df.to_csv(out_path, index=False)


def pre_process(data, system_respose = True, entity_max_length = -1):
    new_data = []
    for line in tqdm(data):
        dialog = json.loads(line)
        # if len(dialog['rec']) == 0:
        #     continue
        # if len(dialog['context']) == 1 and dialog['context'][0] == '':
        #     continue
        context = ''
        for i, utt in enumerate(dialog['context']):
            if utt == '':
                continue
            if i % 2 == 0:
                context += 'User: '
            else:
                context += 'System: '
            context += utt
            context += " <s> "
            # context += " "

        ### if the context is blank. Then we skip.
        if context == '':
            context = '<PAD>'
        ### only consider system response cases.
        # if system_respose:
        #     if i % 2 == 0:
        #         resp = dialog['resp']
        #     else:
        #         resp = ""
        # ### if resp is blank. Then we skip.
        # if resp == "":
        #     continue
        # for item in dialog['rec']:
        example = {
            'conv_id': dialog['conv_id'],
            'context': context,
            # 'un_mask_resp': dialog['un_mask_utt'],
            'resp': dialog['resp'],
            'entity': dialog['entity'],
            'rec': dialog['rec'],
        }
        new_data.append(example)
     
    assert len(data) == len(new_data)
    return new_data

def create_retrieval_unsupervised_training_data(json_file_path, out_path):
    data = read_json_file(json_file_path)
    data = pre_process(data)
    with open(out_path, 'w') as f:
        for line in data:
            context = line['context']
            f.write(context + '\n')

def pre_process_vricr(data, system_respose = True, entity_max_length = -1, bert_tokenizer = None):
    new_data = []
    idx = 0
    for line in tqdm(data):
        dialog = line
        # if len(dialog['rec']) == 0:
        #     continue
        # if len(dialog['context']) == 1 and dialog['context'][0] == '':
        #     continue
        # context = bert_tokenizer.decode(bert_tokenizer.convert_tokens_to_ids(dialog['context_tokens']))
        
        context = ' '.join(dialog['context_tokens'])
        context = context.replace('[SEP]','')
        resp = ' '.join(dialog['response_word'])

        ### if the context is blank. Then we skip.
        if context == '':
            context = '<PAD>'
        ### only consider system response cases.
        # if system_respose:
        #     if i % 2 == 0:
        #         resp = dialog['resp']
        #     else:
        #         resp = ""
        # ### if resp is blank. Then we skip.
        # if resp == "":
        #     continue
        # for item in dialog['rec']:
        conv_id = dialog['identity'].split('/')[0]
        example = {
            'conv_id': conv_id,
            'context': context,
            # 'un_mask_resp': dialog['un_mask_utt'],
            'resp': resp,
            'entity': dialog['context_entities'],
            'rec': dialog['all_movies'],
        }
        new_data.append(example)
     
    assert len(data) == len(new_data)
    return new_data

def create_retrieval_training_data_vricr(pickle_file_path, out_path):
    
    data = read_pickle_file(pickle_file_path)
    data = pre_process_vricr(data, bert_tokenizer= None)
    new_data = {
        "sent0": [],
        "sent1": []
    }
    for line in data:
        new_data["sent0"].append(line['context'])
        new_data['sent1'].append(line['resp'])
    
    df = pd.DataFrame(new_data)
    df.to_csv(out_path, index=False)

def create_retrieval_unsupervised_training_data_vricr(pickle_file_path, out_path):
    data = read_pickle_file(pickle_file_path)
    data = pre_process_vricr(data, bert_tokenizer = None)
    with open(out_path, 'w') as f:
        for line in data:
            context = line['context']
            f.write(context + '\n')


if __name__=='__main__':

    # train_json_path = "aug_prompt/src/data/inspired/train_data_processed.jsonl"
    # train_out_path = "SimCSE-main/data/train_inspired.csv"
    # train_json_path = "aug_prompt/src/data/redial/train_data_processed.jsonl"
    # train_out_path = "SimCSE-main/data/train.csv"
    # create_retrieval_training_data(train_json_path, train_out_path)

    train_pickle_path = "aug_prompt/src/data/Redial_c2/train_recommend_processed_data_Redial.pkl"
    train_out_path = "SimCSE-main/data/train_vricr.csv"

    create_retrieval_training_data_vricr(train_pickle_path, train_out_path)

    # test_json_path = "aug_prompt/src/data/inspired/test_data_processed.jsonl"
    # test_out_path = "SimCSE-main/data/test_inspired.csv"
    # test_json_path = "aug_prompt/src/data/redial/test_data_processed.jsonl"
    # test_out_path = "SimCSE-main/data/test.csv"

    test_pickle_path = "aug_prompt/src/data/Redial_c2/test_recommend_processed_data_Redial.pkl"
    test_out_path = "SimCSE-main/data/test_vricr.csv"
    create_retrieval_training_data_vricr(test_pickle_path, test_out_path)

    valid_pickle_path = "aug_prompt/src/data/Redial_c2/valid_recommend_processed_data_Redial.pkl"
    valid_out_path = "SimCSE-main/data/valid_vricr.csv"
    # valid_json_path = "aug_prompt/src/data/redial/valid_data_processed.jsonl"
    # valid_out_path = "SimCSE-main/data/valid.csv"
    create_retrieval_training_data_vricr(valid_pickle_path, valid_out_path)

    train_un_sup_out_path = "SimCSE-main/data/train_unsup_vricr.txt"
    create_retrieval_unsupervised_training_data_vricr(train_pickle_path, train_un_sup_out_path)