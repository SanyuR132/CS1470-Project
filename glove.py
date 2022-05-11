import linecache
import numpy as np
import tensorflow as tf


def load_glove_embedding(path_to_glove_file, vocab, aspect_vocab, embedding_dim):
    """
    Convert sentences to indexed 
    :param path_to_glove_file: filepath to the GloVe (find GloVe data at nlp.stanford.edu/projects/glove/)
                                we used glove.840B.300d.txt 
    :param vocab:  dictionary, word --> unique index
    :param embedding_dim : integer, number of dimensions each word has in the embedding, 
                            depends on which GloVe dataset is used
    :return: embedding matrix
    """
    embeddings_index = {}
    num_tokens = len(vocab) + 1
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    embedding_matrix_terms = np.zeros((num_tokens, embedding_dim))
    with open(path_to_glove_file, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
            if word in vocab:
                if len(coefs) != 0:
                    embedding_matrix[vocab[word]] = coefs
            if word in aspect_vocab:
                if len(coefs) != 0:
                    embedding_matrix_terms[aspect_vocab[word]] = coefs

    return embedding_matrix, embedding_matrix_terms
