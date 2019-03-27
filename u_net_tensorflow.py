import tensorflow as tf
import numpy as np
import math

def load_npdataset():
    pretrain_x = np.load("/proj/NIRAL/users/mingzhi/npdata64/pretrain_x_resized.npy")
    pretrain_y = np.load("/proj/NIRAL/users/mingzhi/npdata64/pretrain_y_resized.npy")
    #pretrain_x = np.load("pretrain_x_resized.npy")
    #pretrain_y = np.load("pretrain_y_resized.npy")
    pretrain_y = convert_to_one_hot(pretrain_y, 4)

    new_pretrain_x = np.zeros(shape=(22,64,64,64,1))
    new_pretrain_x[:,:,:,:,0] = pretrain_x[:,:,:,:,0]

    new_pretrain_x -= np.mean(new_pretrain_x, axis=0)
    new_pretrain_x /= np.std(new_pretrain_x, axis=0)

    return new_pretrain_x, pretrain_y


def random_mini_batches(X, Y, mini_batch_size):
    m = X.shape[0]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,]
    shuffled_Y = Y[permutation,]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size,]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size,]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Step 3: Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m,]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m,]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = Y.astype(int)
    Y = np.eye(C)[Y]
    return Y

def create_placeholders(n_D0, n_H0, n_W0, n_C0, num_classes):

    X = tf.placeholder(tf.float32, shape=(None, n_D0, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_D0, n_H0, n_W0, num_classes))

    return X, Y

def initialize_parameters(batch_size, D, H, W, c_in, c_out):
    # block 1
    W11 = tf.get_variable("W11", [3, 3, 3, c_in, 64], initializer=tf.contrib.layers.xavier_initializer())
    W12 = tf.get_variable("W12", [3, 3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    B1 = {
        "W1": W11,
        "W2": W12
    }
    # block 2
    W21 = tf.get_variable("W21", [3, 3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    W22 = tf.get_variable("W22", [3, 3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    B2 = {
        "W1": W21,
        "W2": W22
    }
    # block 3
    W31 = tf.get_variable("W31", [3, 3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    W32 = tf.get_variable("W32", [3, 3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    B3 = {
        "W1": W31,
        "W2": W32
    }
    # block 4
    W41 = tf.get_variable("W41", [3, 3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    W42 = tf.get_variable("W42", [3, 3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    W43 = tf.get_variable("W43", [3, 3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    S4 = tf.constant([batch_size, D, H, W, 256], dtype=tf.int32)
    B4 = {
        "W1": W41,
        "W2": W42,
        "W3": W43,
        "S": S4
    }

    # block 7
    W71 = tf.get_variable("W71", [3, 3, 3, 512, 256], initializer=tf.contrib.layers.xavier_initializer())
    W72 = tf.get_variable("W72", [3, 3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    W73 = tf.get_variable("W73", [2, 2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    S7 = tf.constant([batch_size, D, H, W, 128], dtype=tf.int32)
    B7 = {
        "W1": W71,
        "W2": W72,
        "W3": W73,
        "S": S7
    }
    # block 8
    W81 = tf.get_variable("W81", [3, 3, 3, 256, 128], initializer=tf.contrib.layers.xavier_initializer())
    W82 = tf.get_variable("W82", [3, 3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    W83 = tf.get_variable("W83", [2, 2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    S8  = tf.constant([batch_size, D, H, W, 64], dtype=tf.int32)
    B8 = {
        "W1": W81,
        "W2": W82,
        "W3": W83,
        "S": S8
    }
    # block 9
    W91 = tf.get_variable("W91", [3, 3, 3, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
    W92 = tf.get_variable("W92", [3, 3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    W93 = tf.get_variable("W93", [1, 1, 1, 64, c_out], initializer=tf.contrib.layers.xavier_initializer())
    B9 = {
        "W1": W91,
        "W2": W92,
        "W3": W93
    }
    parameters = [B1, B2, B3, B4, B7, B8, B9]
    return parameters


def block_down_forward_propagation(X, block_parameters):
    W1 = block_parameters['W1']
    W2 = block_parameters['W2']

    Z1 = tf.nn.conv3d(X, W1, strides=[1, 1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)

    Z2 = tf.nn.conv3d(A1, W2, strides=[1, 1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)

    return A2

def block_up_forward_propagation(X, block_parameters):
    W1 = block_parameters['W1']
    W2 = block_parameters['W2']
    W3 = block_parameters['W3']
    S = block_parameters['S']

    Z1 = tf.nn.conv3d(X, W1, strides=[1, 1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)

    Z2 = tf.nn.conv3d(A1, W2, strides=[1, 1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)

    Z3 = tf.nn.conv3d_transpose(A2, W3, S, strides=[1, 1, 1, 1, 1], padding="SAME")

    return Z3

def block_last_forward_propagation(X, block_parameters):
    W1 = block_parameters['W1']
    W2 = block_parameters['W2']
    W3 = block_parameters['W3']

    Z1 = tf.nn.conv3d(X, W1, strides=[1, 1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)

    Z2 = tf.nn.conv3d(A1, W2, strides=[1, 1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)

    Z3 = tf.nn.conv3d(A2, W3, strides=[1, 1, 1, 1, 1], padding="SAME")
    A3 = tf.nn.relu(Z3)

    return A3

def forward_proppagation(X, parameters):
    [B1, B2, B3, B4, B5, B6, B7] = parameters

    A0 = X

    A1 = block_down_forward_propagation(A0, B1)
    P1 = tf.nn.max_pool3d(A1, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding="SAME")
    N1 = tf.nn.dropout(P1, keep_prob=0.5)

    A2 = block_down_forward_propagation(N1, B2)
    P2 = tf.nn.max_pool3d(A2, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding="SAME")
    N2 = tf.nn.dropout(P2, keep_prob=0.5)

    A3 = block_down_forward_propagation(N2, B3)
    P3 = tf.nn.max_pool3d(A3, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding="SAME")
    N3 = tf.nn.dropout(P3, keep_prob=0.5)

    A4 = block_up_forward_propagation(N3, B4)
    P4 = tf.concat([A4, A3], -1)
    N4 = tf.nn.dropout(P4, keep_prob=0.5)

    A5 = block_up_forward_propagation(N4, B5)
    P5 = tf.concat([A5, A2], -1)
    N5 = tf.nn.dropout(P5, keep_prob=0.5)

    A6 = block_up_forward_propagation(N5, B6)
    P6 = tf.concat([A6, A1], -1)
    N6 = tf.nn.dropout(P6, keep_prob=0.5)

    A7 = block_last_forward_propagation(N6, B7)

    return A7

def compute_cost(A, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=A, labels=Y))

    return cost

def Dice(pred,true,labels=[0,1,2,3]):
    l = len(labels)
    ret = tf.Variable(tf.zeros(l, tf.float32))
    for i in range(l):
        lab = labels[i]
        p = tf.equal(pred,lab)
        t = tf.equal(true,lab)
        a = tf.reduce_sum(tf.cast(tf.logical_and(p,t), "float"))
        b = tf.reduce_sum(tf.cast(p, "float"))
        c = tf.reduce_sum(tf.cast(t, "float"))
        ret[i].assign(2*(a/(b+c)))
    return ret

costs = []

pretrain_x, pretrain_y = load_npdataset()

minibatch_size = 1
learning_rate = 0.1
m = pretrain_x.shape[0]
c_in = 1
c_out = 4
D = 64
H = 64
W = 64
tf.reset_default_graph()
X, Y = create_placeholders(D, H, W, c_in, c_out)
parameters = initialize_parameters(minibatch_size, D, H, W, c_in, c_out)
A = forward_proppagation(X,parameters)

cost = compute_cost(A,Y)
dice = Dice(A,Y)

#correct_prediction = tf.equal(A, Y)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(3):

        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(pretrain_x, pretrain_y, minibatch_size)
        minibatch_costs = []
        minibatch_accuracies = []

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch
            _, minibatch_train_dice, minibatch_train_cost = sess.run([optimizer, dice, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            print(minibatch_train_dice)
            minibatch_costs.append(minibatch_train_cost)
            #minibatch_accuracies.append(minibatch_train_accuracy)

        epoch_cost = sum(minibatch_costs) / len(minibatch_costs)
        #epoch_accuracy = sum(minibatch_accuracies) / len(minibatch_accuracies)

        costs.append(epoch_cost)
        #accuracies.append(epoch_accuracy)

        print("*****************************************************************")

        #print("epoch cost detail: " + str(minibatch_costs))
        #print("epoch accuracy detail: " + str(minibatch_accuracies))


        print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        #print(dice)
        #print("Average accuracy after epoch %i: %f" % (epoch, epoch_accuracy))

        print("*****************************************************************")

        #if epoch % 5 == 0:
            #learning_rate = learning_rate * 0.5

    print("Costs: "+str(costs))
    #print("Accuracies: " + str(accuracies))
    saver.save(sess, "model_3.ckpt")