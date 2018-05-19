import numpy as np
import tensorflow as tf

NUM_DIGITS = 10

def binary_encode(i, num_digits):
  return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
  if   i % 15 == 0: return np.array([0, 0, 0, 1])
  elif i % 5  == 0: return np.array([0, 0, 1, 0])
  elif i % 3  == 0: return np.array([0, 1, 0, 0])
  else:             return np.array([1, 0, 0, 0])

x_train = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
y_train = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

def init_weights(shape):
  return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(x, w_h, w_o):
  h = tf.nn.relu(tf.matmul(x, w_h))
  return tf.matmul(h, w_o)

x = tf.placeholder("float", [None, NUM_DIGITS])
y = tf.placeholder("float", [None, 4])

NUM_HIDDEN = 100

w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

yy = model(x, w_h, w_o)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yy))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

predict_op = tf.argmax(yy, 1)

def fizz_buzz(i, prediction):
  return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

BATCH_SIZE = 128

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(10001):
  p = np.random.permutation(range(len(x_train)))
  x_train, y_train = x_train[p], y_train[p]

  for start in range(0, len(x_train), BATCH_SIZE):
    end = start + BATCH_SIZE
    sess.run(train_op, feed_dict={x: x_train[start:end], y: y_train[start:end]})

  if epoch % 1000 == 0:
    print(epoch, np.mean(np.argmax(y_train, 1) == sess.run(predict_op, feed_dict={x: x_train, y: y_train})))

numbers = np.arange(1, 101)
x_test = np.transpose(binary_encode(numbers, NUM_DIGITS))
y_test = sess.run(predict_op, feed_dict={x: x_test})
output = np.vectorize(fizz_buzz)(numbers, y_test)

print(output)
