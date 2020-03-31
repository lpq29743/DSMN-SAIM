import re
import xml.etree.ElementTree as ET


def xml2txt(file_name, save_name):
    texts, aspect_nums, aspect_infos = [], [], []
    train_tree = ET.parse(file_name)
    train_root = train_tree.getroot()
    for sentence in train_root:
        text = sentence.find('text').text
        aspect_num = 0
        aspect_info = {}
        for asp_terms in sentence.iter('aspectTerms'):
            for asp_term in asp_terms.findall('aspectTerm'):
                if asp_term.get('polarity') == 'conflict':
                    continue
                from_id = int(asp_term.get('from'))
                to_id = int(asp_term.get('to'))
                text = text[:from_id] + '￥' * (to_id - from_id) + text[to_id:]
                aspect = asp_term.get('term')
                polarity = asp_term.get('polarity')
                aspect_info[from_id] = (aspect, polarity)
                aspect_num += 1

        if aspect_num == 0:
            continue
        text = re.sub('[￥]+', 'aspect_term', text)
        texts.append(text)
        aspect_nums.append(aspect_num)
        aspect_infos.append(aspect_info)

    # 存储至文件
    # 存储格式：
    # 第一行句子，第二行对象词数目
    # 以下为对象词，每个对象词包括对象内容，对象位置（通过aspect_term来判断位置），以及对象标签
    with open(save_name, 'w') as f:
        for text, aspect_num, aspect_info in zip(texts, aspect_nums, aspect_infos):
            f.write('%s\n%s\n' % (text, aspect_num))
            aspect_info = dict(sorted(aspect_info.items()))
            for _, v in aspect_info.items():
                aspect, polarity = v
                f.write('%s\n%s\n' % (aspect, polarity))


xml2txt('data/restaurant/train.xml', 'data/restaurant/train.txt')
