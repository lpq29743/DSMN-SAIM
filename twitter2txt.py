import re
import xml.etree.ElementTree as ET


def xml2txt(file_name, save_name):
    texts, aspect_nums, aspect_infos = [], [], []
    f = open(file_name, 'r')
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        sentence = lines[i].strip()
        aspect = lines[i + 1].strip()
        polarity = lines[i + 2].strip()
        new_sentence = sentence.replace('$T$', 'aspect_term')
        texts.append(new_sentence)
        aspect_nums.append(1)
        aspect_infos.append((aspect, polarity))

    # 存储至文件
    # 存储格式：
    # 第一行句子，第二行对象词数目
    # 以下为对象词，每个对象词包括对象内容，对象位置（通过aspect_term来判断位置），以及对象标签
    with open(save_name, 'w') as f:
        for text, aspect_num, aspect_info in zip(texts, aspect_nums, aspect_infos):
            f.write('%s\n%s\n' % (text, aspect_num))
            aspect, polarity = aspect_info
            if polarity == '-1':
                polarity = 'negative'
            elif polarity == '0':
                polarity = 'neutral'
            else:
                polarity = 'positive'
            f.write('%s\n%s\n' % (aspect, polarity))


xml2txt('data/Twitter/test_data.txt', 'data/Twitter/test.txt')
