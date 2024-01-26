import json
import os

import torch
from loguru import logger
from collections import defaultdict


class DBpedia:
    def __init__(self, dataset, debug=True):
        self.debug = debug

        self.dataset_dir = os.path.join('data', dataset)
        with open(os.path.join(self.dataset_dir, 'dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        self._process_entity_kg()

    def _process_entity_kg(self, SELF_LOOP_ID = 185):

        topic2id = self.entity2id
        id2entity = {idx: entity for entity, idx in topic2id.items()}
        n_entity = len(topic2id)
        edge_list = []
        entity2neighbor = defaultdict(list)  # {entityId: List[entity]}

        for entity in range(n_entity+1):
            edge_list.append((entity, entity, SELF_LOOP_ID))
            if str(entity) not in self.entity_kg:
                continue
            for tail_and_relation in self.entity_kg[str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:  # and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt = defaultdict(int)
        relation_idx = {}
        for h,t, r  in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000 and r not in relation_idx:
                relation_idx[r] = len(relation_idx)+1
        edge_list = [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000]

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(relation_idx)
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {self.num_entities}, #item: {len(self.item_ids)}'
            )

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            'item_ids': self.item_ids,
        }
        return kg_info
