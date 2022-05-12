import numpy as np
import tensorflow as tf
import re
import spacy
import xml.etree.ElementTree as ET

PAD_TOKEN = "*PAD*"
UNK_TOKEN = "*UNK*"

spacy_en = spacy.load('en_core_web_sm')


def read_data(file_name, ATSA=False):
    tree = ET.parse(file_name)
    sentences = tree.getroot()
    labels = []
    aspect_list = []
    sentence_data = []
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        text = text.lower()
        aspects = None
        if ATSA:
            aspects = sentence.find('aspectTerms')
        else:
            aspects = sentence.find('aspectCategories')
        if aspects is None:
            continue
        for aspect in aspects:
            if ATSA:
                aspect_name = aspect.get('term')
            else:
                aspect_name = aspect.get('category')
            polarity = aspect.get('polarity')
            sentence_data.append(text)
            aspect_list.append(aspect_name)
            labels.append(polarity)
    return sentence_data, aspect_list, labels


def sentence_tokenizer(sentence):
    sent = spacy_en(sentence)
    return [word.text for word in sent]


def build_vocab(sentences):
    """
Builds vocab from list of sentences
    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
"""
    tokens = []
    for s in sentences:
        tokens.extend(s)
    all_words = sorted(list(set([PAD_TOKEN, UNK_TOKEN] + tokens)))

    vocab = {word: i for i, word in enumerate(all_words)}

    return vocab, vocab[PAD_TOKEN]


def convert_to_id(vocab, sentences):
    """
Convert sentences to indexed
    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
"""
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def pad_sentence(sentence, num_padding):
    if (num_padding % 2 == 0):
        num_left = num_right = num_padding // 2
    else:
        num_left = int(num_padding / 2)
        num_right = num_left + 1
    sentence = [PAD_TOKEN]*num_left + \
        sentence + [PAD_TOKEN]*num_right
    return sentence


def get_label_ids(train_labels, test_labels):
    label_dict = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
        'conflict': 3
    }
    train_labels = [label_dict[label] for label in train_labels]
    test_labels = [label_dict[label] for label in test_labels]
    return train_labels, test_labels


def get_aspect_categories_ids(train_aspects, test_aspects):
    aspect_categories_dict = {
        'food': 0,
        'service': 1,
        'staff': 2,
        'price': 3,
        'ambience': 4,
        'menu': 5,
        'place': 6,
        'miscellaneous': 7
    }
    train_aspects = [aspect_categories_dict[aspect]
                     for aspect in train_aspects]
    test_aspects = [aspect_categories_dict[aspect]
                    for aspect in test_aspects]
    return train_aspects, test_aspects


def get_data(train_data_file, test_data_file, batch_size, ATSA):
    train_sents, train_aspects, train_labels = read_data(train_data_file, ATSA)
    test_sents, test_aspects, test_labels = read_data(test_data_file, ATSA)
    max_sent_len_train = 0
    max_sent_len_test = 0
    max_term_len_ATSA_train = 0
    max_term_len_ATSA_test = 0

    max_sent = None

    tokenized_train = []
    for sent in train_sents:
        tokenized_sent = sentence_tokenizer(sent)
        if len(tokenized_sent) > max_sent_len_train:
            max_sent_len_train = len(tokenized_sent)
        tokenized_train.append(tokenized_sent)

    print(
        f'in preprocess: max_sent_len (in training data) = {max_sent_len_train}')
    print(f'longest sentence: {max_sent}')

    tokenized_test = []
    for sent in test_sents:
        tokenized_sent = sentence_tokenizer(sent)
        if len(tokenized_sent) > max_sent_len_test:
            max_sent_len_test = len(tokenized_sent)
        tokenized_test.append(tokenized_sent)

    vocab, token_id = build_vocab(tokenized_train)

    tokenized_train = [pad_sentence(
        sent, max_sent_len_train - len(sent)) for sent in tokenized_train]

    tokenized_test = [pad_sentence(
        sent, max_sent_len_train - len(sent)) for sent in tokenized_test]

    train_ids = convert_to_id(vocab, tokenized_train)
    test_ids = convert_to_id(vocab, tokenized_test)

    train_labels, test_labels = get_label_ids(train_labels, test_labels)

    if not ATSA:
        train_aspects, test_aspects = get_aspect_categories_ids(
            train_aspects, test_aspects)
        term_vocab = {
            'food': 0,
            'service': 1,
            'staff': 2,
            'price': 3,
            'ambience': 4,
            'menu': 5,
            'place': 6,
            'miscellaneous': 7
        }
    else:
        tokenized_term_train = []
        tokenized_term_test = []
        for term in train_aspects:
            tokenized_term = sentence_tokenizer(term)
            if len(tokenized_term) > max_term_len_ATSA_train:
                max_term_len_ATSA_train = len(tokenized_term)
            tokenized_term_train.append(tokenized_term)
        # TODO TODO TODO check if padding should be on both sides vs just one side TODO TODO TODO
        tokenized_term_train = [pad_sentence(
            term, max_term_len_ATSA_train - len(term)) for term in tokenized_term_train]
        for term in test_aspects:
            tokenized_term = sentence_tokenizer(term)
            if len(term) > max_term_len_ATSA_test:
                max_term_len_ATSA_test = len(tokenized_term)
            tokenized_term_test.append(tokenized_term)
        # TODO TODO TODO check if padding should be on both sides vs just one side TODO TODO TODO
        tokenized_term_test = [pad_sentence(
            term, max_term_len_ATSA_train - len(term)) for term in tokenized_term_test]

        term_vocab, term_token_id = build_vocab(tokenized_term_train)
        train_aspects = convert_to_id(term_vocab, tokenized_term_train)
        test_aspects = convert_to_id(term_vocab, tokenized_term_test)

    train_loader = tf.data.Dataset.from_tensor_slices(
        (train_ids, train_aspects, train_labels))
    train_loader = train_loader.shuffle(buffer_size=len(train_loader))
    train_loader = train_loader.batch(batch_size)

    test_loader = tf.data.Dataset.from_tensor_slices(
        (test_ids, test_aspects, test_labels))
    test_loader = test_loader.batch(batch_size)

    return train_loader, test_loader, vocab, term_vocab
