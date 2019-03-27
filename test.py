import tensorflow as tf
import u_net_tensorflow

pretrain_x, pretrain_y = u_net_tensorflow.load_npdataset()
minibatch_size = 1
m = pretrain_x.shape[0]
c_in = 2
c_out = 4
D = 63
H = 75
W = 50
tf.reset_default_graph()
X, Y = u_net_tensorflow.create_placeholders(D, H, W, c_in, c_out)
parameters = u_net_tensorflow.initialize_parameters(minibatch_size, D, H, W, c_in, c_out)
A = u_net_tensorflow.forward_proppagation(X, parameters)

minibatch_accuracies = []

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt.meta")

    minibatches = u_net_tensorflow.random_mini_batches(pretrain_x, pretrain_y, minibatch_size)

    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        correct_prediction = tf.equal(A, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        minibatch_train_accuracy = accuracy.eval({X: minibatch_X, Y: minibatch_Y})
        minibatch_accuracies.append(minibatch_train_accuracy)

    train_accuracy = sum(minibatch_accuracies)/len(minibatch_accuracies)
    print("Train Accuracy:", train_accuracy)