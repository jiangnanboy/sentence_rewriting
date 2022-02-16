import jieba.analyse
import jieba.posseg
from typing import List

def extract_keywords_tfidf(sentence, pos=('n', 'nr', 'ns'), topK=10) -> List:
    '''
    tfidf for extract keyword
    :param sentence:
    :param pos:
    :param topK:
    :return:
    '''
    keywords = jieba.analyse.extract_tags(sentence, topK=topK, allowPOS=pos)
    return keywords

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

