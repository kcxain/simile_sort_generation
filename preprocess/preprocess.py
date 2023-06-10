import random
import json
from random import sample

random.seed(221)

import Levenshtein
import tqdm

num_simile = 3
# 正例 >> 喻体->喻体的上位词 >> 喻体->随机替换/属性->随机替换 >> 喻体->随机替换, 属性->随机替换
ALL_SENTENCE_TYPES = [
    'positive',  # 0 -> 0
    'vehicle_isa_replace',  # 1 -> 1
    'vehicle_random_replace',  # 2 -> 2
    'property_random_replace',  # 3 -> 2
    'both_random_replace'  # 4 -> 3
]

ALL_SIMILE_TYPES = [
    'positive',  # 0
    'vehicle_isa_replace',  # 1
    'single_random_replace',  # 2
    # 'property_random_replace',
    'both_random_replace'  # 3
]

CONCEPT_NET_ROOT = '../data/negative-vehicle-property_0524/'
POSITIVE_DATA = CONCEPT_NET_ROOT + 'SPGC_parsed.txt'
VEHICLE_IS_A_DATA = CONCEPT_NET_ROOT + 'SPGC_isa_0530.txt'
INFO_DATA = '../data/data_info.json'


def load_from_json(data_path: str) -> list:
    """Loads simile data from json file.
    """
    try:
        with open(data_path, 'r') as f:
            dialog_data = json.load(f)
    except json.JSONDecodeError:
        # print('Fail to load {} with json.load().'.format(data_path))
        # print('Try to load it line by line with json.loads()...')
        with open(data_path, 'r') as f:
            dialog_data = [json.loads(line) for line in f.readlines()]
    return dialog_data


def random_word(l, w, num):
    l_sample = None
    while l_sample is None or w in l_sample:
        l_sample = sample(l, num)
    return l_sample


def process_data():
    with open(INFO_DATA, 'r') as f:
        data_info = json.load(f)
    d = {}
    all_topics = data_info['topic']
    all_vehicles = data_info['vehicle']
    all_properties = data_info['property']
    all_sentences = []
    positive_data = read_data(POSITIVE_DATA)
    for line in positive_data:
        topic = line[1]
        property = line[2]
        vehicle = line[3]
        sentence_mask_property = line[0]
        sentence = line[0].replace('_', property)
        # sentence_mask_vehicle = sentence.replace(vehicle, '_')
        sentence_property_split = line[0].split('_')
        sentence_property_split[-1] = sentence_property_split[-1].replace(vehicle, '_', 1)
        sentence_mask_vehicle = property.join(sentence_property_split)
        # print(sentence_mask_vehicle)
        # input('>')
        sentence_key = sentence
        random_vehicle = random_word(all_vehicles, vehicle, num_simile)
        random_property = random_word(all_properties, property, num_simile)
        # print(sentence_key)
        # input('>')
        d[sentence_key] = {'positive': [sentence],
                           'vehicle_isa_replace': [],
                           'vehicle_random_replace': [sentence_mask_vehicle.replace('_', w) for w in random_vehicle],
                           'property_random_replace': [sentence_mask_property.replace('_', w) for w in random_property],
                           'both_random_replace': [sentence.replace(vehicle, w_v).replace(property, w_p) for w_v, w_p in
                                                   zip(random_vehicle, random_property)]
                           }
    isa_data = read_data(VEHICLE_IS_A_DATA)
    all_vehicle_isa_replace = []
    for line in tqdm.tqdm(isa_data):
        topic = line[1]
        property = line[2]
        vehicle = line[3]
        sentence = line[0]
        # print(sentence)
        # sentence_as_split = sentence.split(' as ')
        # sentence_as_split[-1] = sentence_as_split[-1].replace(vehicle, '_')
        # sentence_mask_vehicle = ' as '.join(sentence_as_split)]
        key = find_min_key(sentence, d.keys())
        # print(sentence)
        # print(key)
        # input('>')
        if 'vehicle_isa_replace' in d[key]:
            d[key]['vehicle_isa_replace'].append(sentence)
        else:
            d[key]['vehicle_isa_replace'] = [sentence]
        all_vehicle_isa_replace.append(sentence)
    write_json('./SPGC_data.json', d)
    write_four_level(d, all_vehicle_isa_replace)


def write_four_level(d, all_vehicle_isa_replace):
    new_d = {}
    for key in d.keys():
        vehicle_random_replace = d[key]['vehicle_random_replace']
        property_random_replace = d[key]['property_random_replace']
        single_random_replace = vehicle_random_replace + property_random_replace
        single_random_replace = random.sample(single_random_replace, num_simile)
        # TODO 没有 isa 怎么办？
        vehicle_isa_replace = d[key]['vehicle_isa_replace']
        if len(vehicle_isa_replace) > num_simile:
            vehicle_isa_replace = random.sample(d[key]['vehicle_isa_replace'], num_simile)
        else:
            vehicle_isa_replace.append(random.sample(all_vehicle_isa_replace, num_simile - len(vehicle_isa_replace)))
        # ALL_SIMILE_TYPES = [
        #     'positive',  # 0
        #     'vehicle_isa_replace',  # 1
        #     'single_random_replace',  # 2
        #     # 'property_random_replace',
        #     'both_random_replace'  # 3
        # ]
        new_d[key] = {
            'positive': d[key]['positive'] * num_simile,
            'vehicle_isa_replace': vehicle_isa_replace,
            'single_random_replace': single_random_replace,
            'both_random_replace': d[key]['both_random_replace'],
        }
    write_split_json('./dataset', new_d)


def find_min_key(sentence, d_keys):
    min_distance = 10000
    min_key = None
    for key in d_keys:
        cur_distance = Levenshtein.distance(sentence, key)
        if cur_distance < min_distance:
            min_distance = cur_distance
            min_key = key

    assert min_key is not None

    return min_key


def write_json(path, d):
    with open(path, 'w') as f:
        for key in d.keys():
            json.dump(d[key], f)
            f.write('\n')


def write_split_json(path, d):
    with open(path + '/train.json', 'w') as f_train:
        with open(path + '/test.json', 'w') as f_test:
            with open(path + '/dev.json', 'w') as f_dev:
                for key in d.keys():
                    n = random.random()
                    if n < 0.1:
                        json.dump(d[key], f_dev)
                        f_dev.write('\n')
                    elif n < 0.2:
                        json.dump(d[key], f_test)
                        f_test.write('\n')
                    else:
                        json.dump(d[key], f_train)
                        f_train.write('\n')


def read_data(data_dir, max_row=None):
    dataset = []
    with open(data_dir, 'r', encoding="utf-8") as f:
        idx = 0
        for data in f.readlines():
            if max_row is not None:
                idx += 1
                if idx > max_row:
                    break
            sentence, topic, property, vehicle = data.replace('\n', '').split('\t')
            dataset.append([sentence, topic, property, vehicle])
    f.close()
    return dataset


def main():
    dataset = read_data(POSITIVE_DATA)
    topics = set()
    properties = set()
    vehicles = set()
    for line in dataset:
        topics.add(line[1])
        properties.add(line[2])
        vehicles.add(line[3])
    info = {
        'topic': list(topics),
        'property': list(properties),
        'vehicle': list(vehicles)
    }
    with open('./data_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False)


if __name__ == '__main__':
    process_data()
    # main()
