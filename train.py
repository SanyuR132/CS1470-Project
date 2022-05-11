import os
import sys
import tensorflow as tf
import numpy as np

def train(args, train_iter, dev_iter, mixed_test_iter, model, text_field, aspect_field, sml_field, predict_iter):
  time_stamps = []

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  steps = 0
  # model.train()
  dev_acc, mixed_acc = 0, 0
  for epoch in range(1, args.epochs + 1):
    for batch in train_iter:
      feature, aspect, target = batch.text, batch.aspect, batch.sentiment

      # feature.data.t_()
      # if len(feature) < 2:
      #   continue
      # if not args.aspect_phrase:
      #   aspect.data.unsqueeze_(0)
      # # aspect.data.t_() not sure what this is
      # target.data.sub_(1)

      with tf.GradientTape() as tape:
        logit, _, _ = model(feature, aspect)
        loss = tf.nn.softmax_cross_entropy_with_logits(target, logit)

      gradients = tape.gradient(loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      steps += 1
      if steps % 100 == 0:
        corrects = (tf.reduce_max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch.batch_size
        print(f'Batch {steps} - loss: {accuracy} acc: {accuracy} corrects: {corrects} batch_size{batch.batch_size}')
    if epoch == args.epochs:
      dev_acc, _, _ = test(dev_iter, model, args)
      if mixed_test_iter:
        mixed_acc, _, _ = test(mixed_test_iter, model, args)
      else:
        mixed_acc = 0.0
  return (dev_acc, mixed_acc), time_stamps

  
def test(data_iter, model, args):
  # model.eval()
  corrects, avg_loss = 0, 0
  loss = None
  for batch in data_iter:
    feature, aspect, target = batch.text, batch.aspect, batch.sentiment

    feature.data.t_()
    if not args.aspect_phrase:
      aspect.data.unsqueeze_(0)
      target.data.sub_(1)
    
    logit, pooling_input, relu_weights = model(feature, aspect)
    loss = tf.nn.softmax_cross_entropy_with_logits(target, logit)
    avg_loss += loss.data[0]
    correct += (tf.reduce_max(logit, 1)[1].view(target.size()).data == target.data).sum()
        
  size = len(data_iter.dataset)
  avg_loss = loss.data[0]/size
  accuracy = 100.0 * corrects/size
  model.train()
  return accuracy, pooling_input, relu_weights
