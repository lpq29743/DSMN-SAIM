import os
import ast
import spacy
import numpy as np
import pandas as pd
from errno import ENOENT
from collections import Counter

nlp = spacy.load("en_core_web_sm")


def get_data_info(train_fname, test_fname, save_fname, pre_processed):
    word2id, max_sentence_len, max_aspect_len, max_aspect_num = {}, 0, 0, 0
    word2id['<pad>'] = 0
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        with open(save_fname, 'r') as f:
            for line in f:
                content = line.strip().split()
                if len(content) == 4:
                    max_sentence_len = int(content[1])
                    max_aspect_len = int(content[2])
                    max_aspect_num = int(content[3])
                else:
                    word2id[content[0]] = int(content[1])
    else:
        if not os.path.isfile(train_fname):
            raise IOError(ENOENT, 'Not a file', train_fname)
        if not os.path.isfile(test_fname):
            raise IOError(ENOENT, 'Not a file', test_fname)

        words = []

        train_f = open(train_fname, 'r')
        while True:
            line = train_f.readline()
            if not line:
                break
            sentence = line.strip()
            aspect_num = int(train_f.readline().strip())
            sptoks = nlp(sentence)
            if len(sptoks) > max_sentence_len:
                max_sentence_len = len(sptoks)
            words.extend([sp.text.lower() for sp in sptoks])
            if aspect_num > max_aspect_num:
                max_aspect_num = aspect_num
            for _ in range(aspect_num):
                aspect = train_f.readline().strip()
                train_f.readline()
                t_sptoks = nlp(aspect)
                if len(t_sptoks) > max_aspect_len:
                    max_aspect_len = len(t_sptoks)
                words.extend([sp.text.lower() for sp in t_sptoks])

        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and str(word).strip() != '':
                word2id[word] = len(word2id)

        test_f = open(test_fname, 'r')
        while True:
            line = test_f.readline()
            if not line:
                break
            sentence = line.strip()
            aspect_num = int(test_f.readline().strip())
            sptoks = nlp(sentence)
            if len(sptoks) > max_sentence_len:
                max_sentence_len = len(sptoks)
            words.extend([sp.text.lower() for sp in sptoks])
            if aspect_num > max_aspect_num:
                max_aspect_num = aspect_num
            for _ in range(aspect_num):
                aspect = test_f.readline().strip()
                test_f.readline()
                t_sptoks = nlp(aspect)
                if len(t_sptoks) > max_aspect_len:
                    max_aspect_len = len(t_sptoks)
                words.extend([sp.text.lower() for sp in t_sptoks])

        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and str(word).strip() != '':
                word2id[word] = len(word2id)

        with open(save_fname, 'w') as f:
            f.write('length %s %s %s\n' % (max_sentence_len, max_aspect_len, max_aspect_num))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))

    print(
        'There are %s words in the dataset, the max length of sentence is %s, the max length of aspect is %s, and the max num of aspect is %s' % (
            len(word2id), max_sentence_len, max_aspect_len, max_aspect_num))
    return word2id, max_sentence_len, max_aspect_len, max_aspect_num


class Token:
    def __init__(self, id, text, head, child):
        self.id = id
        self.text = text
        self.head = head
        self.child = child


def get_mask_sem(sentence, max_sentence_len, aspect_pos, max_aspect_num):
    doc = nlp(sentence)

    mask = []
    token_list = []
    is_visited = [0 for _ in range(len(doc))]
    asp_info = [1 for _ in range(len(aspect_pos))]
    sen_info = [1 for _ in range(len(doc))]
    for token in doc:
        token_list.append(Token(token.i, token.text, token.head.i, [child.i for child in token.children]))
    tmp_mask = [0 for _ in range(len(doc))]
    for i in range(len(doc)):
        if 'aspect_word' in token_list[i].text:
            for j, pos in enumerate(aspect_pos):
                if pos == i:
                    asp_info[j] = 0
            tmp_mask[i] = 1
            tmp_mask[token_list[i].head] = 1
            for k in token_list[i].child:
                tmp_mask[k] = 1
            is_visited[i] = 1
            sen_info[i] = 0
    mask.append(tmp_mask + [0] * (max_sentence_len - len(tmp_mask)))
    for i in range(1, 10):
        for j, asp in enumerate(aspect_pos):
            if tmp_mask[asp] == 0:
                asp_info[j] += 1
        for j in range(len(doc)):
            if tmp_mask[j] == 0:
                sen_info[j] += 1
        tmp_mask = mask[-1].copy()
        ids = []
        for j in range(len(doc)):
            if is_visited[j] == 0 and tmp_mask[j] == 1:
                ids.append(j)
        for j in ids:
            tmp_mask[token_list[j].head] = 1
            for k in token_list[j].child:
                tmp_mask[k] = 1
            is_visited[j] = 1
        mask.append(tmp_mask + [0] * (max_sentence_len - len(tmp_mask)))
    asp_loc = [1 - inf / 10.0 for inf in asp_info]
    asp_loc = asp_loc + [0.0] * (max_aspect_num - len(asp_loc))
    sen_loc = [1 - inf / 10.0 for inf in sen_info]
    sen_loc = sen_loc + [0.0] * (max_sentence_len - len(sen_loc))
    return mask, asp_loc, sen_loc


def get_mask_pos(sentence, max_sentence_len, aspect_pos, max_aspect_num):
    doc = nlp(sentence)

    mask = []
    is_visited = [0 for _ in range(len(doc))]
    asp_info = [1 for _ in range(len(aspect_pos))]
    sen_info = [1 for _ in range(len(doc))]
    tmp_mask = [0 for _ in range(len(doc))]
    for i, d in enumerate(doc):
        if d.text.lower() == 'aspect_word':
            for j, pos in enumerate(aspect_pos):
                if pos == i:
                    asp_info[j] = 0
            tmp_mask[i] = 1
            if i - 1 >= 0:
                tmp_mask[i - 1] = 1
            if i + 1 < len(doc):
                tmp_mask[i + 1] = 1
            is_visited[i] = 1
            sen_info[i] = 0
    mask.append(tmp_mask + [0] * (max_sentence_len - len(tmp_mask)))
    for i in range(1, 10):
        for j, asp in enumerate(aspect_pos):
            if tmp_mask[asp] == 0:
                asp_info[j] += 1
        for j in range(len(doc)):
            if tmp_mask[j] == 0:
                sen_info[j] += 1
        tmp_mask = mask[-1].copy()
        ids = []
        for j in range(len(doc)):
            if is_visited[j] == 0 and tmp_mask[j] == 1:
                ids.append(j)
        for j in ids:
            if j - 1 >= 0:
                tmp_mask[j - 1] = 1
            if j + 1 < len(doc):
                tmp_mask[j + 1] = 1
            is_visited[j] = 1
        mask.append(tmp_mask + [0] * (max_sentence_len - len(tmp_mask)))
    asp_loc = [1 - inf / 10.0 for inf in asp_info]
    asp_loc = asp_loc + [0.0] * (max_aspect_num - len(asp_loc))
    sen_loc = [1 - inf / 10.0 for inf in sen_info]
    sen_loc = sen_loc + [0.0] * (max_sentence_len - len(sen_loc))
    return mask, asp_loc, sen_loc


def get_asp_moment(cur_polarities, asp_loc):
    l = len(cur_polarities)

    mean_sum = 0.0
    for i in range(l):
        # mean_sum += cur_polarities[i] * asp_loc[i]
        mean_sum += cur_polarities[i]
    mean = mean_sum / l

    var_sum = 0.0
    for i in range(l):
        var_sum += (mean - cur_polarities[i]) ** 2
        # var_sum += asp_loc[i] * ((mean - cur_polarities[i]) ** 2)
    var = var_sum / l
    return [(mean + 1) / 2.0, 1 - (mean + 1) / 2.0], [var, 1 - var]


def read_data(fname, word2id, max_sentence_len, max_aspect_len, max_aspect_num, save_fname, select_method,
              pre_processed):
    sentences, sentence_lens, num, aspects, aspect_lens, sentence_infos, sentence_locs, aspect_locs, aspect_means, aspect_vars, labels = [], [], [], [], [], [], [], [], [], [], []
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 11):
            sentences.append(ast.literal_eval(lines[i]))
            sentence_lens.append(ast.literal_eval(lines[i + 1]))
            num.append(ast.literal_eval(lines[i + 2]))
            aspects.append(ast.literal_eval(lines[i + 3]))
            aspect_lens.append(ast.literal_eval(lines[i + 4]))
            sentence_infos.append(ast.literal_eval(lines[i + 5]))
            sentence_locs.append(ast.literal_eval(lines[i + 6]))
            aspect_locs.append(ast.literal_eval(lines[i + 7]))
            aspect_means.append(ast.literal_eval(lines[i + 8]))
            aspect_vars.append(ast.literal_eval(lines[i + 9]))
            labels.append(ast.literal_eval(lines[i + 10]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)

        with open(fname, 'r') as f, open(save_fname, 'w') as sf:
            while True:
                line = f.readline()
                if not line:
                    break
                sentence = line.strip()
                aspect_num = int(f.readline().strip())
                cur_aspects = []
                cur_polarities = []
                cur_labels = []
                for _ in range(aspect_num):
                    cur_aspect = f.readline().strip()
                    cur_aspects.append(cur_aspect)
                    cur_label = f.readline().strip()
                    if cur_label == 'negative':
                        cur_polarities.append(-1.0)
                    elif cur_label == 'neutral':
                        cur_polarities.append(0.0)
                    elif cur_label == "positive":
                        cur_polarities.append(1.0)
                    cur_labels.append(cur_label)

                sptoks = nlp(sentence)
                aspect_pos = []
                if len(sptoks.text.strip()) != 0:
                    cnt = 0
                    ids = []
                    for sptok in sptoks:
                        if sptok.text.lower() in word2id:
                            ids.append(word2id[sptok.text.lower()])
                        if sptok.text == 'aspect_term':
                            aspect_pos.append(cnt)
                        cnt += 1

                sentences.append(ids + [0] * (max_sentence_len - len(ids)))
                sf.write("%s\n" % sentences[-1])
                sentence_lens.append(len(sptoks))
                sf.write("%s\n" % sentence_lens[-1])
                num.append(len(cur_polarities))
                sf.write("%s\n" % num[-1])

                aspects_tmp, aspect_lens_tmp, sentence_infos_tmp, sentence_locs_tmp, aspect_locs_tmp, aspect_means_tmp, aspect_vars_tmp, labels_tmp = [], [], [], [], [], [], [], []
                cnt = 0
                for aspect, label in zip(cur_aspects, cur_labels):
                    cnt += 1
                    if 'Twitter' not in fname:
                        groups = sentence.split('aspect_term')
                        nth_split = ['aspect_term'.join(groups[:cnt]), 'aspect_term'.join(groups[cnt:])]
                        dp_text = 'aspect_word'.join(nth_split)
                    else:
                        dp_text = sentence.replace('aspect_term', 'aspect_word')

                    t_sptoks = nlp(aspect)
                    t_ids = []
                    for sptok in t_sptoks:
                        if sptok.text.lower() in word2id:
                            t_ids.append(word2id[sptok.text.lower()])
                    aspects_tmp.append(t_ids + [0] * (max_aspect_len - len(t_ids)))
                    aspect_lens_tmp.append(len(t_sptoks))

                    if 'Twitter' in fname:
                        aspect_pos = [aspect_pos[0]]
                    if select_method == 'pos':
                        mask, asp_loc, sen_loc = get_mask_pos(dp_text, max_sentence_len, aspect_pos, max_aspect_num)
                    elif select_method == 'sem':
                        mask, asp_loc, sen_loc = get_mask_sem(dp_text, max_sentence_len, aspect_pos, max_aspect_num)
                    sentence_infos_tmp.append(mask)
                    sentence_locs_tmp.append(sen_loc)
                    aspect_locs_tmp.append(asp_loc)

                    asp_mean, asp_var = get_asp_moment(cur_polarities, asp_loc)
                    aspect_means_tmp.append(asp_mean)
                    aspect_vars_tmp.append(asp_var)

                    if label == 'negative':
                        labels_tmp.append([1, 0, 0])
                    elif label == 'neutral':
                        labels_tmp.append([0, 1, 0])
                    elif label == "positive":
                        labels_tmp.append([0, 0, 1])

                aspects.append(aspects_tmp + [[0] * max_aspect_len] * (max_aspect_num - len(aspects_tmp)))
                sf.write("%s\n" % aspects[-1])
                aspect_lens.append(aspect_lens_tmp + [0] * (max_aspect_num - len(aspect_lens_tmp)))
                sf.write("%s\n" % aspect_lens[-1])
                sentence_infos.append(
                    sentence_infos_tmp + [[[0] * max_sentence_len] * 10] * (max_aspect_num - len(sentence_infos_tmp)))
                sf.write("%s\n" % sentence_infos[-1])
                sentence_locs.append(
                    sentence_locs_tmp + [[0] * max_sentence_len] * (max_aspect_num - len(sentence_locs_tmp)))
                sf.write("%s\n" % sentence_locs[-1])
                aspect_locs.append(aspect_locs_tmp + [[0] * max_aspect_num] * (max_aspect_num - len(aspect_locs_tmp)))
                sf.write("%s\n" % aspect_locs[-1])
                aspect_means.append(aspect_means_tmp + [[0, 0]] * (max_aspect_num - len(aspect_means_tmp)))
                sf.write("%s\n" % aspect_means[-1])
                aspect_vars.append(aspect_vars_tmp + [[0, 0]] * (max_aspect_num - len(aspect_vars_tmp)))
                sf.write("%s\n" % aspect_vars[-1])
                labels.append(labels_tmp + [[0] * 3] * (max_aspect_num - len(labels_tmp)))
                sf.write("%s\n" % labels[-1])

    print("Read %s sentences from %s" % (len(sentences), fname))
    return np.asarray(sentences), np.asarray(sentence_lens), np.asarray(num), np.asarray(aspects), np.asarray(
        aspect_lens), np.asarray(sentence_infos), np.asarray(sentence_locs), np.asarray(aspect_locs), np.asarray(
        aspect_means), np.asarray(aspect_vars), np.asarray(labels)


def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    np.random.seed(0)
    word2vec = np.random.uniform(-0.25, 0.25, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                oov = oov - 1
    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec


def save_analysis_result(test_fname, best_y_pred, analysis_fname):
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    test_f = open(test_fname, 'r')
    analysis_result = []
    i = 0
    while True:
        line = test_f.readline()
        if not line:
            break
        sentence = line.strip()
        aspect_num = int(test_f.readline().strip())
        for _ in range(aspect_num):
            aspect = test_f.readline().strip()
            ground_truth = test_f.readline().strip()
            analysis_result.append((sentence, aspect, ground_truth, id2label[best_y_pred[i]],
                                    ground_truth == id2label[best_y_pred[i]]))
            i += 1
    df = pd.DataFrame(analysis_result)
    df.columns = ["sentence", "aspect", "ground_truth", "predict_label", "result"]
    df.to_csv(analysis_fname)
    assert i == len(best_y_pred)


def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.seed(0)
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size:(i + 1) * batch_size]


def get_batch_data(data, batch_size, is_shuffle):
    sentences, sentence_lens, num, aspects, aspect_lens, sentence_infos, sentence_locs, aspect_locs, aspect_means, aspect_vars, labels = data
    for index in get_batch_index(len(sentences), batch_size, is_shuffle):
        feed_dict = {
            'sentences': sentences[index],
            'sentence_lens': sentence_lens[index],
            'num': num[index],
            'aspects': aspects[index],
            'aspect_lens': aspect_lens[index],
            'sentence_infos': sentence_infos[index],
            'sentence_locs': sentence_locs[index],
            'aspect_locs': aspect_locs[index],
            'aspect_means': aspect_means[index],
            'aspect_vars': aspect_vars[index],
            'labels': labels[index],
        }
        yield feed_dict, len(index)
