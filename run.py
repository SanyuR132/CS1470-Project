import argparse
import CNN_atsa
import CNN_gate_aspect
from preprocess import *
import os
import time
import tensorflow as tf

# https://github.com/jiangqn/GCAE-pytorch/blob/master/main.py

embedding_size = 300
num_epochs = 15

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=300)

args = parser.parse_args()
# maybe just hard code some values instead of having a parser

def main():
    train_loader, test_loader = get_data(train_data_file="./data/train.xml", test_data_file="./data/test.xml", batch_size=100, ATSA=False)

    model = CNN_atsa.CNN_Gate_Aspect_Text()
    model2 = CNN_gate_aspect.CNN_Gate_Aspect_Text()
    acc = train(model, train_loader)
    print(f'acc is {acc}')
    test_acc = test(model, test_loader)
    print(f'test acc is {test_acc}')


def train(model, train_loader):
    for epoch in range(num_epochs):
        train_total_cases = 0
        train_correct_cases = 0
        for data in train_loader:
            sentences, aspects, labels = data
            with tf.GradientTape() as tape:
                logit, x, y = model.forward(sentences, aspects)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logit)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
            predictions = tf.reduce_max(logit, 1)
            train_total_cases += labels.shape[0]
            train_correct_cases += (predictions == labels).sum().item()
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