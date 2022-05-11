import argparse

from numpy import dtype
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
num_epochs = 1

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=300)

args = parser.parse_args()
# maybe just hard code some values instead of having a parser


def pickle_it():
    acsa_train_loader, acsa_test_loader, acsa_vocab, acsa_aspect_vocab = get_data(
        train_data_file="./data/acsa_train.xml", test_data_file="./data/acsa_test.xml", batch_size=100, ATSA=False)
    # atsa_train_loader, atsa_test_loader, atsa_vocab, atsa_aspect_vocab = get_data(train_data_file="./data/atsa_train.xml", test_data_file="./data/atsa_test.xml", batch_size=100, ATSA=True)
    print('finished loading and beginning embedding')
    acsa_embedding_matrix, acsa_embedding_matrix_aspect = load_glove_embedding(path_to_glove_file='D:\Glove\glove.840B.300d.txt', vocab=acsa_vocab, aspect_vocab=acsa_aspect_vocab, embedding_dim=300)
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
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(num_epochs):
        print(f'current epoch = {epoch}')
        train_total_cases = 0
        train_correct_cases = 0
        count = 0
        max_correct = 0
        for data in train_loader:
            sentences, aspects, labels = data
            if len(sentences) != 100:
                continue
            with tf.GradientTape() as tape:
                logit, x, y = model.forward(sentences, aspects)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logit)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            predictions = tf.math.argmax(logit, 1, output_type=tf.int32)
            train_total_cases += labels.shape[0]
            filtered = len(tf.where(labels > 3))
            if filtered > 0:
                print(f'there were {filtered} values over 2')
            num_correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
            print(f'acc is {num_correct / 100}')
            max_correct = max(max_correct, num_correct)
            train_correct_cases += num_correct
            count += 1
        train_accuracy = train_correct_cases / train_total_cases
        print(f'epoch {epoch} has acc {train_accuracy}, max_correct is {max_correct}')
    return train_accuracy



def test(model, test_loader):
    for epoch in range(num_epochs):
        test_total_cases = 0
        test_correct_cases = 0
        max_correct = 0
        count = 0
        for data in test_loader:
            sentences, aspects, labels = data
            if len(sentences) != 100:
                continue
            print(f'sentences shape is {sentences.shape}')
            print(f'aspects shape is {aspects.shape}')
            print(f'labels shape is {labels.shape}')
            logit, x, y = model.forward(sentences, aspects)

            predictions = tf.math.argmax(logit, 1, output_type=tf.int32)
            test_total_cases += labels.shape[0]
            num_correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
            test_correct_cases += num_correct
            max_correct = max(max_correct, num_correct)
            count += 1
        test_accuracy = test_correct_cases / test_total_cases
        print(f'epoch {epoch} has acc {test_accuracy}, max_correct is {max_correct}')
    return test_accuracy


if __name__ == '__main__':
    main()
