#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Mecthew on 2019-12-02
import os
from keras.preprocessing import sequence
import numpy as np
from modeling.constant import NGRAM_MAX_LENGTH, NGRAM_MIN_LENGTH, ROOT_DIR, DEBUG
from integration.dataset import AESDataset
from integration.ingestion import _parse_args
from modeling.constant import PROMPT_NUM, IS_CROSS_DATASET, MAX_SEQ_LEN
from integration.metrics import kappa
from modeling.models import *
import random
import csv


def pad_seq(data, pad_len):
    return sequence.pad_sequences(data, maxlen=pad_len, dtype='float32', padding='post', truncating='post')


def save_predicts(all_predicts):
    fout = open(ROOT_DIR + '/result_output/predicts-' + time.strftime("%Y-%m-%d@%H-%M-%S") + '.tsv',
                'w', encoding='utf8')
    args = _parse_args()
    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=IS_CROSS_DATASET)
    essay_list, essay_id_list, essay_set_list = dataset.get_test()
    for i in range(PROMPT_NUM):
        essay_ids = essay_id_list[i]
        essay_sets = essay_set_list[i]
        for idx, prediction in enumerate(all_predicts[i]):
            fout.write(f'{essay_ids[idx]}\t{essay_sets[idx]}\t{prediction}\n')
    fout.close()


# Save essays for computing kernel string
def prompt_write_into_txt(essays_of_prompt, prompt, data_type=None):
    hisk_dir = ROOT_DIR + '/essay_data/HISK'
    hisk_input_dir = os.path.join(hisk_dir, 'input')
    os.makedirs(hisk_dir, exist_ok=True)
    os.makedirs(hisk_input_dir, exist_ok=True)

    hisk_input_file = os.path.join(hisk_input_dir, get_prompt_str(prompt=prompt, data_type=data_type))
    if not os.path.exists(hisk_input_file):
        with open(hisk_input_file, 'w', encoding='utf8') as fout:
            for essay in essays_of_prompt:
                fout.write(essay.replace('\n', '').strip() + '\n')


def get_prompt_str(prompt, data_type=None):
    if data_type:
        return "prompt@" + str(prompt) + '-' + data_type + '.txt'
    else:
        return "prompt@" + str(prompt) + '.txt'


def execute_compute_string_kernel_command(kernel_type, input_file_1, input_file_2, output_file):
    kernel_dir = ROOT_DIR + '/resource/String_Kernels_Package_v1.0/String_Kernels_Package/code/'

    if input_file_2 is None:
        string_kernel_command = (
            f'java -classpath {kernel_dir} '
            f'ComputeStringKernel '
            f'{kernel_type} {NGRAM_MIN_LENGTH} {NGRAM_MAX_LENGTH} '
            f'{input_file_1} {output_file}'
        )
    else:
        string_kernel_command = (
            f'java -classpath {kernel_dir} '
            f'ComputeStringKernel '
            f'{kernel_type} {NGRAM_MIN_LENGTH} {NGRAM_MAX_LENGTH} '
            f'{input_file_1} {input_file_2} {output_file}'
        )
    os.system(string_kernel_command)


def compute_string_kernel():
    args = _parse_args()
    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=IS_CROSS_DATASET)
    x_train_list, y_train_list = dataset.get_train()
    x_dev_list, y_dev_list = dataset.get_dev()
    essay_list, essay_id_list, essay_set_list = dataset.get_test()

    # train_len = sum(list(map(len, x_train_list)))
    # dev_len = sum(list(map(len, x_dev_list)))
    # test_len = sum(list(map(len, essay_list)))
    # print(train_len, dev_len, test_len)

    for idx in range(PROMPT_NUM):
        x_train, y_train = x_train_list[idx], y_train_list[idx]
        x_dev, y_dev = x_dev_list[idx], y_dev_list[idx]
        essay_test, essay_id_test, essay_set_test = essay_list[idx], essay_id_list[idx], essay_set_list[idx]
        all_prompt_essays = np.concatenate([x_train, x_dev, essay_test], axis=0)
        prompt_write_into_txt(essays_of_prompt=all_prompt_essays, prompt=idx+1)

    hisk_input_dir = ROOT_DIR + '/essay_data/HISK/input'
    hisk_output_dir = ROOT_DIR + '/essay_data/HISK/output1'
    os.makedirs(hisk_output_dir, exist_ok=True)
    # Compute kernel string matrix
    for idx in range(PROMPT_NUM):
        # kernel_type can be "intersection", "presence" or "spectrum"
        essays_input = os.path.join(hisk_input_dir, get_prompt_str(prompt=idx+1))
        essays_output = os.path.join(hisk_output_dir, get_prompt_str(prompt=idx+1))
        execute_compute_string_kernel_command(kernel_type='intersection',
                                              input_file_1=essays_input,
                                              input_file_2=None,
                                              output_file=essays_output)

        '''
        execute_compute_string_kernel_command(kernel_type='intersection',
                                              input_file_1=train_input,
                                              input_file_2=None,
                                              output_file=train_output)
        execute_compute_string_kernel_command(kernel_type='intersection',
                                              input_file_1=dev_input,
                                              input_file_2=train_input,
                                              output_file=dev_output)
        execute_compute_string_kernel_command(kernel_type='intersection',
                                              input_file_1=test_input,
                                              input_file_2=train_input,
                                              output_file=test_output)
        '''


'''
    Use neural network
'''
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
import spacy
import language_check
import string
import pandas as pd
import re
from modeling.constant import TEXT_DIM
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def correct_language(essay_list, save_path=None):
    tool = language_check.LanguageTool('en-US')
    matches = [tool.check(essay) for essay in essay_list]
    corrects = [language_check.correct(essay, match) for essay, match in zip(essay_list, matches)]
    if save_path:
        fout = open(save_path, 'w', encoding='utf8')
        for essay in corrects:
            fout.write(essay.strip() + '\n')
        fout.close()
    return corrects


def correct_essays(save_path=None):
    args = _parse_args()
    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=True)
    x_train_list, y_train_list = dataset.get_train()
    x_dev_list, y_dev_list = dataset.get_dev()
    essay_list, essay_id_list, essay_set_list = dataset.get_test()
    all_essays = np.concatenate((x_train_list, x_dev_list, essay_list), axis=0)
    corrects = correct_language(essay_list=all_essays, save_path=save_path)
    return corrects


# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_essays(essays, logging=False):
    punctuations = string.punctuation
    stop_words = set(stopwords.words('english'))
    nlp = spacy.load('en_core_web_sm')
    texts = []
    counter = 1
    for essay in essays:
        essay = str(essay)
        if counter % 2000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(essays)))
        counter += 1
        essay = nlp(essay, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in essay if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return texts


# Define function to preprocess text for a word2vec model
def cleanup_essay_word2vec(essays, logging=False):
    sentences = []
    counter = 1
    nlp = spacy.load('en_core_web_sm')
    for essay in essays:
        essay = str(essay)
        if counter % 2000 == 0 and logging:
            print("Processed %d out of %d documents" % (counter, len(essays)))
        # Disable tagger so that lemma_ of personal pronouns (I, me, etc) don't getted marked as "-PRON-"
        essay = nlp(essay, disable=['tagger'])
        # Grab lemmatized form of words and make lowercase
        essay = " ".join([tok.lemma_.lower() for tok in essay])
        # Split into sentences based on punctuation
        essay = re.split(r"[.?!;] ", essay)
        # Remove commas, periods, and other punctuation (mostly commas)
        essay = [re.sub(r"[.,;:!?\"]", "", sent) for sent in essay]
        # Split into words
        essay = [sent.split() for sent in essay]
        sentences += essay
        counter += 1
    return sentences


def generate_word2vec(text_dim, model_save_path):
    corrects_path = ROOT_DIR + '/essay_data/corrects/all_essays.txt'
    os.makedirs(path=ROOT_DIR + '/essay_data/corrects', exist_ok=True)
    if not os.path.exists(corrects_path):
        correct_essays(save_path=corrects_path)

    fin = open(corrects_path, 'r', encoding='utf8')
    all_essays = []
    for line in fin:
        all_essays.append(line.strip())
    cleaned_texts = cleanup_essay_word2vec(all_essays, logging=True)

    wordvec_model = Word2Vec(cleaned_texts, size=text_dim, window=5, min_count=3, workers=4, sg=1)
    wordvec_model.save(model_save_path)


def embedding_predicts(wordvec_dict):
    args = _parse_args()
    dataset = AESDataset(args.dataset_dir, prompt_num=PROMPT_NUM, is_cross_dataset=IS_CROSS_DATASET, use_correct=True)
    x_train_list, y_train_list = dataset.get_train()
    x_dev_list, y_dev_list = dataset.get_dev()
    essay_list, essay_id_list, essay_set_list = dataset.get_test()

    cleaned_dir = ROOT_DIR + '/essay_data/cleaned'
    cleaned_path = os.path.join(cleaned_dir, 'cleaned.txt')
    os.makedirs(cleaned_dir, exist_ok=True)

    if IS_CROSS_DATASET:
        x_train_cleaned = cleanup_essays(x_train_list, logging=True)
        x_dev_cleaned = cleanup_essays(x_dev_list, logging=True)
        x_test_cleaned = cleanup_essays(essay_list, logging=True)
    else:
        if not os.path.exists(cleaned_path):
            x_train_cleaned = [cleanup_essays(x_train_list[i], logging=True) for i in range(PROMPT_NUM)]
            x_dev_cleaned = [cleanup_essays(x_dev_list[i], logging=True) for i in range(PROMPT_NUM)]
            x_test_cleaned = [cleanup_essays(essay_list[i], logging=True) for i in range(PROMPT_NUM)]
            fout = open(cleaned_path, 'w', encoding='utf8')
            for i in range(PROMPT_NUM):
                fout.write('\n'.join(x_train_cleaned[i]) + '\n')
                fout.write('\n'.join(x_dev_cleaned[i]) + '\n')
                fout.write('\n'.join(x_test_cleaned[i]) + '\n')
            fout.close()
        else:
            x_train_cleaned, x_dev_cleaned, x_test_cleaned = [], [], []
            begin_idx = 0
            with open(cleaned_path, 'r', encoding='utf8') as fin:
                cleaned_essays = [line.strip() for line in fin]
            for prompt_i in range(PROMPT_NUM):
                x_train_cleaned.append(cleaned_essays[begin_idx:begin_idx+len(x_train_list[prompt_i])])
                begin_idx += len(x_train_list[prompt_i])
                x_dev_cleaned.append(cleaned_essays[begin_idx:begin_idx+len(x_dev_list[prompt_i])])
                begin_idx += len(x_dev_list[prompt_i])
                x_test_cleaned.append(cleaned_essays[begin_idx:begin_idx+len(essay_list[prompt_i])])
                begin_idx += len(essay_list[prompt_i])

        prompt_cnt = 0
        k_list = []
        use_regression = True
        model_lib = {
            # LSTM_MODEL: Lstm,
            # CNN_MODEL: Cnn,
            CNN_MULTIPLE: CnnMulInputs,
            LSTM_MULTIPLE: LstmMulInputs,
            # CRNN_MODEL: crnn
        }
        repeat_num = 6
        prompt_predicts = []
        for i in range(0, PROMPT_NUM):
            prompt_cnt += 1
            x_train_vec = np.array([create_average_vec(essay, text_dim=TEXT_DIM, wordvec_dict=wordvec_dict)
                           for essay in x_train_cleaned[i]])
            x_dev_vec = np.array([create_average_vec(essay, text_dim=TEXT_DIM, wordvec_dict=wordvec_dict)
                           for essay in x_dev_cleaned[i]])
            x_test_vec = np.array([create_average_vec(essay, text_dim=TEXT_DIM, wordvec_dict=wordvec_dict)
                           for essay in x_test_cleaned[i]])

            x_train_seq_vec = np.array([create_sequence_vec(essay, text_dim=TEXT_DIM, wordvec_dict=wordvec_dict)
                                        for essay in x_train_cleaned[i]])
            x_dev_seq_vec = np.array([create_sequence_vec(essay, text_dim=TEXT_DIM, wordvec_dict=wordvec_dict)
                                      for essay in x_dev_cleaned[i]])
            x_test_seq_vec = np.array([create_sequence_vec(essay, text_dim=TEXT_DIM, wordvec_dict=wordvec_dict)
                                       for essay in x_test_cleaned[i]])

            y_train = y_train_list[i]
            y_dev = y_dev_list[i]
            max_class, min_class = max(y_train), min(y_train)
            if use_regression:
                output_dim = 1
            else:
                output_dim = max_class + 1
            hisk_dir = ROOT_DIR + '/essay_data/HISK/output'
            hisk_all_dir = ROOT_DIR + '/essay_data/HISK/output-all'
            hisk_all = [np.array(line.strip().split()).astype(int) for line
                        in open(hisk_all_dir + '/prompt@' + str(i+1) + '.txt', 'r', encoding='utf8')]
            hisk_train = [np.array(line.strip().split()).astype(int) for line
                          in open(hisk_dir+'/prompt@' + str(i+1) + '-train.txt', 'r', encoding='utf8')]
            hisk_dev = [np.array(line.strip().split()).astype(int) for line
                          in open(hisk_dir+'/prompt@' + str(i+1) + '-dev.txt', 'r', encoding='utf8')]
            hisk_test = [np.array(line.strip().split()).astype(int) for line
                          in open(hisk_dir+'/prompt@' + str(i+1) + '-test.txt', 'r', encoding='utf8')]
            hisk_train, hisk_dev, hisk_test = np.array(hisk_train), np.array(hisk_dev), np.array(hisk_test)

            sscalar = StandardScaler()
            hisk_all = sscalar.fit_transform(hisk_all)
            hisk_train, hisk_dev, hisk_test = np.array(hisk_all[:len(y_train)]), np.array(hisk_all[len(y_train):len(y_train)+len(y_dev)]),\
                                              np.array(hisk_all[-len(essay_list[i]):])

            x_train_vec = np.concatenate([x_train_vec, hisk_train], axis=-1)
            x_dev_vec = np.concatenate([x_dev_vec, hisk_dev], axis=-1)
            x_test_vec = np.concatenate([x_test_vec, hisk_test], axis=-1)
            x_train_vec = hisk_train
            x_dev_vec = hisk_dev
            x_test_vec = hisk_test

            x_train_vec = x_train_seq_vec
            x_dev_vec = x_dev_seq_vec
            x_test_vec = x_test_seq_vec

            print(f'Prompt@{i+1}, num_classes: {max_class-min_class+1}; '
                  f'x_train shape: {np.array(x_train_vec).shape}, y_train shape: {np.array(y_train).shape}; '
                  f'x_dev shape: {np.array(x_dev_vec).shape}, y_dev shape: {np.array(y_dev).shape}; '
                  f'x_test shape: {np.array(x_test_vec).shape}, y_test shape: {np.array(essay_list[i]).shape}')

            total_predicts = []

            for model_name in model_lib.keys():
                predicts_list = []
                dev_predicts_list = []
                for idx in range(repeat_num):
                    x_train_input = x_train_vec
                    x_dev_input = x_dev_vec
                    x_test_input = x_test_vec
                    my_model = model_lib[model_name]()
                    if 'mul' in model_name:
                        my_model.init_model(prompt=i+1,
                                            input_shape1=x_train_vec.shape[1:], input_shape2=np.array(hisk_train).shape[-1],
                                            output_dim=output_dim)
                        x_train_input = [x_train_vec, hisk_train]
                        x_dev_input = [x_dev_vec, hisk_dev]
                        x_test_input = [x_test_vec, hisk_test]
                    else:
                        my_model.init_model(input_shape=x_train_vec.shape[1:], output_dim=output_dim)
                    my_model.fit(x_train_input, y_train, x_dev_input, y_dev, train_loop_num=1)
                    predicts = np.round(my_model.predict(x_test_input)).reshape(-1, 1)
                    dev_predicts = np.round(my_model.predict(x_dev_input)).reshape(-1, 1)
                    # predicts = mmscaler.inverse_transform(predicts)
                    predicts_list.append(predicts)
                    dev_predicts_list.append(dev_predicts)

                dev_kappa_list = []
                for dev_predict in dev_predicts_list:
                    dev_kappa = kappa(y_true=y_dev, y_pred=dev_predict, weights="quadratic")
                    dev_kappa_list.append(dev_kappa)
                aver_dev_kappa = np.mean(dev_kappa_list)

                cmp_kapaa, cmp_kappa_list = aver_dev_kappa, dev_kappa_list
                selected_list = [predict for predict, kp in zip(predicts_list, cmp_kappa_list) if kp >= cmp_kapaa]

                aver_predicts = np.mean(np.concatenate(selected_list, axis=-1), axis=-1)
                total_predicts.append(aver_predicts.reshape(-1, 1))

            ensemble_predicts = np.mean(np.concatenate(total_predicts, axis=-1), axis=-1)
            prompt_predicts.append(ensemble_predicts)

        os.makedirs(ROOT_DIR + '/result_output', exist_ok=True)
        save_predicts(prompt_predicts)


def create_average_vec(essay, text_dim, wordvec_dict):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    # max_vec = np.zeros((text_dim,), dtype='float32')
    for word in essay.split():
        if word in wordvec_dict.keys():
            average = np.add(average, wordvec_dict[word])
            num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average


def create_sequence_vec(essay, text_dim, wordvec_dict):
    rng = np.random.RandomState(123)
    unknown = np.asarray(rng.normal(size=text_dim, loc=0, scale=0.5))
    padding = np.zeros(shape=(text_dim,))

    sequence_vec = []
    for word in essay.split():
        if word in wordvec_dict.keys():
            sequence_vec.append(np.array(wordvec_dict[word]))
        else:
            sequence_vec.append(unknown)
    padding_len = max(0, MAX_SEQ_LEN - len(essay.split()))
    for i in range(padding_len):
        sequence_vec.append(padding)
    return np.array(sequence_vec[:MAX_SEQ_LEN])


def get_wordvec_dict(wordvec_model):
    wordvec_dict = {}
    for word in wordvec_model.wv.vocab.keys():
        wordvec_dict[word] = wordvec_model.wv[word]
    return wordvec_dict


def get_glove_wordvec(glove_path):
    wordvec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            word = tokens[0]
            vec = np.array(tokens[1:], dtype=np.float)
            wordvec_dict[word] = vec
    return wordvec_dict


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model_save_path = ROOT_DIR + '/resource/wordvec_model'
    glove_path = '/home/chengfeng/embedding/glove/glove.6B.' + str(TEXT_DIM) + 'd.txt'
    wordvec_model = Word2Vec.load(model_save_path)
    # wv_dict = get_glove_wordvec(glove_path)
    wv_dict = get_wordvec_dict(wordvec_model)
    # compute_string_kernel()
    # generate_word2vec(text_dim=300, model_save_path=model_save_path)
    embedding_predicts(wordvec_dict=wv_dict)
