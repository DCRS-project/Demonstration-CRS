from simcse import SimCSE
import pandas as pd
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import pickle

class CustomDataset(Dataset):

    def __init__(self, contexts, responses, raw_data):
        super().__init__()
        self.contexts  = contexts
        self.responses = responses
        self.raw_data = raw_data
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        response = self.responses[idx]
        raw = self.raw_data[idx]
        rec = 0
        if len(raw['rec']) > 0:
            rec = 1

        return context, response, rec

    def collate_fn(self, batch):
        batch_contexts, batch_responses, batch_rec = list(zip(*batch))
        return batch_contexts, batch_responses, batch_rec

model = SimCSE("./result/my-unsup-simcse-bert-base-uncased-vricr")

def read_data(train_path):
    df = pd.read_csv(train_path)
    responses = []
    contexts = []
    for _, row in df.iterrows():
        responses.append(str(row['sent1']))
        if row['sent0'] == '':
            contexts.append(" ")
        else:
            contexts.append(row['sent0'])
    return contexts, responses


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data

def read_jsonl_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(x) for x in lines]
        return lines

def create_retrieval_oritented_data(out_path, train_contexts, train_responses, raw_data, contexts = None, top_k = 10, threshold = 0.1, split = 'train', raw_train_data = None):
    sentences = train_contexts
    model.build_index(sentences)

    if split == 'train':
        with open(out_path, 'w') as f:
            for idx, (context, raw_example) in enumerate(tqdm(list(zip(train_contexts, raw_data)))):
                results = model.search(context, top_k= top_k, threshold= threshold)
                retrieved_responses = []
                retrieved_un_mask_responses = []
                retrieved_contexts = []
                retrieved_context_entities = []
                retrieved_response_entities = []
                j = 0
                new_example = raw_example
                assert len(results) > 0
                while j < len(results) and len(retrieved_responses) < 3:
                    res = results[j]
                    assert len(res) == 3
                    if raw_data[res[1]]['identity'].split('/')[0] == raw_example['identity'].split('/')[0]:
                        j+= 1
                        continue
                    
                    retrieved_responses.append(train_responses[res[1]])
                    retrieved_contexts.append(train_contexts[res[1]])
                    # retrieved_un_mask_responses.append(raw_data[res[1]]['un_mask_utt'])
                    # retrieved_context_entities.extend(raw_data[res[1]]['entity'])
                    # retrieved_response_entities.extend(raw_data[res[1]]['rec'])
                    retrieved_context_entities.extend(raw_data[res[1]]['context_entities'])
                    retrieved_response_entities.extend(raw_data[res[1]]['all_movies'])
                    

                    j +=1
                new_example['retrieved_resp'] = retrieved_responses
                new_example['retrieved_contexts'] = retrieved_contexts
                # new_example['un_mask_retrieved_resp'] = retrieved_un_mask_responses
                new_example['retrieved_context_entity'] = retrieved_context_entities
                new_example['retrieved_response_entity'] = retrieved_response_entities
                f.write(json.dumps(new_example) + '\n')
    else:
         with open(out_path, 'w') as f:
            
            for context, raw_example in list(zip(contexts, raw_data)):
                results = model.search(context, top_k= top_k, threshold= threshold)
                retrieved_responses = []
                retrieved_un_mask_responses = []
                retrieved_contexts = []
                retrieved_context_entities = []
                retrieved_response_entities = []
                j = 0
                new_example = raw_example
                while j < len(results) and len(retrieved_responses) < 3:
                    res = results[j]
                    # assert train_contexts[res[1]].lower() == raw_data[res[i]]['conversation'].lower()
                    retrieved_responses.append(train_responses[res[1]])
                    retrieved_contexts.append(train_contexts[res[1]])
                    # retrieved_un_mask_responses.append(raw_train_data[res[1]]['un_mask_utt'])
                    # retrieved_context_entities.extend(raw_train_data[res[1]]['entity'])
                    # retrieved_response_entities.extend(raw_train_data[res[1]]['rec'])

                    retrieved_context_entities.extend(raw_train_data[res[1]]['context_entities'])
                    retrieved_response_entities.extend(raw_train_data[res[1]]['all_movies'])
                    j +=1
                new_example['retrieved_resp'] = retrieved_responses
                new_example['retrieved_contexts'] = retrieved_contexts
                new_example['un_mask_retrieved_resp'] = retrieved_un_mask_responses
                new_example['retrieved_context_entity'] = retrieved_context_entities
                new_example['retrieved_response_entity'] = retrieved_response_entities
                f.write(json.dumps(new_example) + '\n') 


def compute_recall_metrics(model, test_contexts, test_responses, raw_data, train_contexts, train_responses = None):
    # sentences = train_contexts
    # model.build_index(sentences)
    r1 = 0
    r10 = 0
    r50 = 0
    # num_examples = 20
    # for idx, (context, response) in enumerate(list(zip(test_contexts[:num_examples], test_responses[:num_examples]))):
    #     results = model.search(context, top_k = 3)
    #     print('context: ', context)
    #     print('ground truth: ', response)
    #     for i, res in enumerate(results):
    #         print(f"res_{i}: response: {train_responses[res[1]]}, score: {res[2]} ")
    #     print('-------------------------')
    # return
    
    # dataset = CustomDataset(test_contexts, test_responses, raw_data)
    # data_loader = DataLoader(dataset, batch_size = 64, collate_fn = dataset.collate_fn)
    # count = 0

    for contexts, responses, recs in data_loader:
        results = model.search(list(contexts), top_k = 50)
        for response, result, rec in list(zip(responses, results, recs)):
            result = [x[0] for x in result]

            if response in result[:1]:
                r1 += 1
            if response in result[:10]:
                r10 += 1
            if response in result[:50]:
                r50 +=1
            
            count += 1
        
    r1 = r1 / count
    r10 = r10 / count
    r50 = r50 / count

    return r1, r10, r50, count

def compute_retrieval_based_metrics(file_path):

    r1, r10, r50 = 0, 0, 0
    with open(file_path, 'r') as f:
        lines = f.readlines()

        for example in lines:
            example = json.loads(example)
            rec = example['rec']
            retrieved_resp_items = example['retrieved_resp_entity']

            for item in rec:
                if item in retrieved_resp_items:
                    r1 +=1 
    
    print('recall @1: ', r1/ len(lines))
                


if __name__ == "__main__":

    # train_new_path = '../aug_prompt/src/data/inspired/train_data_processed_retrieval.jsonl'
    # valid_new_path = '../aug_prompt/src/data/inspired/valid_data_processed_retrieval.jsonl'
    # test_new_path = '../aug_prompt/src/data/inspired/test_data_processed_retrieval.jsonl'

    # train_new_path = '../aug_prompt/src/data/redial/train_data_processed_retrieval_pre.jsonl'
    # valid_new_path = '../aug_prompt/src/data/redial/valid_data_processed_retrieval_pre.jsonl'
    # test_new_path = '../aug_prompt/src/data/redial/test_data_processed_retrieval_pre.jsonl'

    train_new_path = '../aug_prompt/src/data/Redial_c2/train_recommend_processed_data_Redial_retrieval.jsonl' 
    valid_new_path = '../aug_prompt/src/data/Redial_c2/valid_recommend_processed_data_Redial_retrieval.jsonl' 
    test_new_path = '../aug_prompt/src/data/Redial_c2/test_recommend_processed_data_Redial_retrieval.jsonl' 

    train_path = 'data/train_vricr.csv'
    test_path = 'data/test_vricr.csv'
    valid_path = 'data/valid_vricr.csv'

    raw_test_path = '../aug_prompt/src/data/Redial_c2/test_recommend_processed_data_Redial.pkl'
    raw_train_path = '../aug_prompt/src/data/Redial_c2/train_recommend_processed_data_Redial.pkl'    
    raw_valid_path = '../aug_prompt/src/data/Redial_c2/valid_recommend_processed_data_Redial.pkl'

    # raw_test_path = '../aug_prompt/src/data/redial/test_data_processed.jsonl'
    # raw_train_path = '../aug_prompt/src/data/redial/train_data_processed.jsonl'    
    # raw_valid_path = '../aug_prompt/src/data/redial/valid_data_processed.jsonl'

    train_contexts, train_responses = read_data(train_path)
    test_contexts, test_responses = read_data(test_path)
    valid_contexts, valid_responses = read_data(valid_path)

    # raw_test_data = read_jsonl_file(raw_test_path)
    # raw_train_data = read_jsonl_file(raw_train_path)
    # raw_valid_data = read_jsonl_file(raw_valid_path)

    # raw_test_data = read_jsonl_file(raw_test_path)
    # raw_train_data = read_jsonl_file(raw_train_path)
    # raw_valid_data = read_jsonl_file(raw_valid_path)

    raw_test_data = read_pickle_file(raw_test_path)
    raw_train_data = read_pickle_file(raw_train_path)
    raw_valid_data = read_pickle_file(raw_valid_path)

    print(len(raw_train_data))

    # print(compute_recall_metrics(model, test_contexts, test_responses, raw_test_data, train_contexts, train_responses))
    create_retrieval_oritented_data(train_new_path, train_contexts, train_responses, raw_train_data)
    create_retrieval_oritented_data(test_new_path, train_contexts, train_responses, raw_test_data, test_contexts, split = 'test', raw_train_data=raw_train_data)
    create_retrieval_oritented_data(valid_new_path, train_contexts, train_responses, raw_valid_data, valid_contexts, split = 'valid', raw_train_data=raw_train_data)
