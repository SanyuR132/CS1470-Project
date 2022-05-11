import argparse
import CNN_atsa
import CNN_gate_aspect
from glove import *
from preprocess import *
import os
import time
import tensorflow as tf
import pickle

# https://github.com/jiangqn/GCAE-pytorch/blob/master/main.py

embedding_size = 300
num_epochs = 15

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=300)

args = parser.parse_args()
# maybe just hard code some values instead of having a parser


def pickle_it():
    acsa_train_loader, acsa_test_loader, acsa_vocab, acsa_aspect_vocab = get_data(
        train_data_file="./data/acsa_train.xml", test_data_file="./data/acsa_test.xml", batch_size=100, ATSA=False)
    # atsa_train_loader, atsa_test_loader, atsa_vocab, atsa_aspect_vocab = get_data(train_data_file="./data/atsa_train.xml", test_data_file="./data/atsa_test.xml", batch_size=100, ATSA=True)
    print('finished loading and beginning embedding')
    acsa_embedding_matrix, acsa_embedding_matrix_aspect = load_glove_embedding(
        path_to_glove_file='/home/srajakum/Downloads/glove.840B.300d.txt', vocab=acsa_vocab, aspect_vocab=acsa_aspect_vocab, embedding_dim=300)
    # atsa_embedding_matrix, atsa_embedding_matrix_aspect = load_glove_embedding(path_to_glove_file=, vocab=atsa_vocab, aspect_vocab=atsa_aspect_vocab, embedding_dim=300)
    print('finished embeddings')
    acsa_model = CNN_gate_aspect.CNN_Gate_Aspect_Text(
        acsa_embedding_matrix, acsa_embedding_matrix_aspect)
    # atsa_model = CNN_atsa.CNN_Gate_Aspect_Text(atsa_embedding_matrix, atsa_embedding_matrix_aspect)
    print('begin training')
    dbfile4 = open('acsa_embedding_matrix', 'ab')
    pickle.dump(acsa_embedding_matrix, dbfile4)
    dbfile4.close()
    dbfile5 = open('acsa_embedding_matrix_aspect', 'ab')
    pickle.dump(acsa_embedding_matrix_aspect, dbfile5)
    dbfile5.close()
    tf.data.experimental.save(acsa_train_loader, 'acsa_train_load')
    tf.data.experimental.save(acsa_test_loader, 'acsa_test_load')


def main():
    # pickle_it()
    print(f'loading in data')
    # atsa_train_loader, atsa_test_loader, atsa_vocab, atsa_aspect_vocab = get_data(train_data_file="./data/atsa_train.xml", test_data_file="./data/atsa_test.xml", batch_size=100, ATSA=True)
    print('finished loading and beginning embedding')
    # atsa_embedding_matrix, atsa_embedding_matrix_aspect = load_glove_embedding(path_to_glove_file=, vocab=atsa_vocab, aspect_vocab=atsa_aspect_vocab, embedding_dim=300)
    print('finished embeddings')
    acsa_train_loader = tf.data.experimental.load('acsa_train_load')
    acsa_test_loader = tf.data.experimental.load('acsa_test_load')
    dbfile4 = open('acsa_embedding_matrix', 'rb')
    acsa_embedding_matrix = pickle.load(dbfile4)
    dbfile4.close()
    dbfile5 = open('acsa_embedding_matrix_aspect', 'rb')
    acsa_embedding_matrix_aspect = pickle.load(dbfile5)
    dbfile5.close()

    acsa_model = CNN_gate_aspect.CNN_Gate_Aspect_Text(
        acsa_embedding_matrix, acsa_embedding_matrix_aspect)
    # atsa_model = CNN_atsa.CNN_Gate_Aspect_Text(atsa_embedding_matrix, atsa_embedding_matrix_aspect)
    print('begin training')
    acc = train(acsa_model, acsa_train_loader)
    print('finish training')
    print(f'acc is {acc}')
    test_acc = test(acsa_model, acsa_test_loader)
    print(f'test acc is {test_acc}')


def train(model, train_loader):
    for epoch in range(num_epochs):
        print(f'current epoch = {epoch}')
        train_total_cases = 0
        train_correct_cases = 0
        count = 0
        for data in train_loader:
            print(count)
            sentences, aspects, labels = data
            with tf.GradientTape() as tape:
                logit, x, y = model.forward(sentences, aspects)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logit)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            predictions = tf.reduce_max(logit, 1)
            train_total_cases += labels.shape[0]
            train_correct_cases += (predictions == labels).sum().item()
            count += 1
        train_accuracy = train_correct_cases / train_total_cases
        print(f'epoch {epoch} has acc {train_accuracy}')
    return train_accuracy


def test(model, test_loader):
    for epoch in range(num_epochs):
        test_total_cases = 0
        test_correct_cases = 0
        for data in test_loader:
            sentences, aspects, labels = data
            logit, x, y = model.forward(sentences, aspects)

            predictions = tf.reduce_max(logit, 1)
            test_total_cases += labels.shape[0]
            test_correct_cases += (predictions == labels).sum().item()
        test_accuracy = test_correct_cases / test_total_cases
        max_acc = max(max_acc, test_accuracy)
        print(f'epoch {epoch} max accuracy is {max_acc}')
    return test_accuracy


if __name__ == '__main__':
    main()
