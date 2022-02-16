import json
import os
import csv
import logging
import jieba.analyse
from typing import List

def extract_keywords_textrank(sentence, pos=('n', 'nr', 'ns'), topK=10) -> List:
    '''
    textrank for extract keyword
    :param sentence:
    :param pos:
    :param topK:
    :return:
    '''
    keywords = jieba.analyse.textrank(sentence, topK=topK, allowPOS=pos)
    return keywords

def read_json(json_path, csv_path):
    '''
    process json to csv
    :param json_path:
    :param csv_path:
    :return:
    '''
    with open(json_path, 'r', encoding='utf8') as f_read, open(csv_path, 'w', encoding="utf8", newline='') as f_write:
        logging.info('data processing...')
        csv_write = csv.writer(f_write)
        # csv_write.writerow(['title', 'keywords', 'text'])
        for line in f_read:
            line_context = []
            set_ner = set()
            data = json.loads(line)
            title = data.get("title")
            content = data.get("content")
            content_ner = data.get("content_ner")
            if title is None or content is None or content_ner is None:
                continue
            if len(title) == 0 or len(content) == 0 or len(content_ner) == 0:
                continue
            for content_json in content_ner:
                set_ner.add(content_json["value"].strip())
            if len(set_ner) > 0:
                line_context.append(title)
                line_context.append(' '.join(set_ner))
                line_context.append(content)
                csv_write.writerow(line_context)
        logging.info("data processed!")

def build_trainingset(raw_path, save_path):
    print('build training set for aiwirter...')
    with open(raw_path, 'r', encoding='utf8') as f_read, open(save_path, 'w', encoding='utf8', newline='') as f_write:
        csv_write = csv.writer(f_write)
        count = 0
        for line in f_read:
            line = line.strip()
            if len(line) == 0:
                continue
            kw = extract_keywords_textrank(line)
            if len(kw) == 0:
                continue
            count += 1
            kw_content = []
            kw_content.append(' '.join(kw))
            kw_content.append(line)
            csv_write.writerow(kw_content)
            if count == 100000:
                break
    print('process done!')

