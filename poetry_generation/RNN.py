import tensorflow as tf
import numpy as np
from textloader import TextLoader
import matplotlib.pyplot as plt
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
import logging


logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')
logger = logging.getLogger('word_temperature_sampling')
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
fh = logging.FileHandler('word_temp_sampling.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Logging initialized")

# make a list of all the words to choose starting words later 
with open("input.txt", 'r') as f:
    lines = f.readlines()
    data = [word for x in [line.decode('utf-8').split(' ') for line in lines] for word in x]
    print len(lines)
    # batch_size = len(lines)//10

# Global variables
batch_size = 1000
sequence_length = 10 # 50
state_dim = 128
num_layers = 2

data_loader = TextLoader( ".", batch_size, sequence_length )
vocab_size = data_loader.vocab_size  # dimension of one-hot encodings

tf.reset_default_graph()

# ==================================================================

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( in_onehot, sequence_length, axis=1 )
inputs = [tf.squeeze(i, [1]) for i in inputs] # inputs is list length sequence_length; each element is [batch_size, vocab_size]
targets = tf.split( targ_ph, sequence_length, axis=1, ) # targets is list length sequence_length; each element of targets is 1D vector length batch_size

# ------------------
# COMPUTATION GRAPH

with tf.variable_scope("RNN") as scope:
    basic1 = BasicLSTMCell(state_dim)
    basic2 = BasicLSTMCell(state_dim)
    rnn = MultiRNNCell([basic1,basic2])
    initial_state = rnn.zero_state(batch_size, tf.float32) # initial_state is a list of tensors
    output_list, final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, rnn)
    
    init = tf.contrib.layers.variance_scaling_initializer()
    W = tf.Variable(tf.random_normal([state_dim, vocab_size], stddev=0.02))
    b = tf.Variable(tf.random_normal([vocab_size], stddev=0.01))
    logits = [tf.matmul(output, W) + b for output in output_list]

    weights = [1.]*sequence_length
    loss = tf.contrib.legacy_seq2seq.sequence_loss(logits, targets, weights)

    c = 0.001
    optim = tf.train.AdamOptimizer(learning_rate=c).minimize(loss)
    
# ------------------
# SAMPLER GRAPH

# reuse the parameters of the cell and the parameters you used to transform state outputs to logits!

def temperature_sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))  

batch_size = 1
sequence_length = 1
s_in_ph = tf.placeholder( tf.int32, [batch_size], name='inputs' )
s_in_onehot = tf.one_hot( s_in_ph, vocab_size, name="input_onehot" )

s_inputs = tf.split( s_in_onehot, sequence_length, axis=0 )

with tf.variable_scope("Sampler") as scope:
    s_initial_state = rnn.zero_state(batch_size, tf.float32) # initial_state is a list of tensors

    s_output_list, s_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(s_inputs, s_initial_state, rnn)
    
    s_logits = [tf.matmul(output, W) + b for output in s_output_list] # W, b from previous RNN
    # s_probs = tf.nn.softmax(s_logits)
    # s_probs = temperature_sample(s_logits)

# ==================================================================
# ==================================================================

def sample( num=200, prime='ab' ):
    s_state = sess.run( s_initial_state )

    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    ret = prime
    char = prime[-1]
    for n in xrange(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        # ops = [s_probs]
        # ops.extend( list(s_final_state) )
        ops = [s_logits]
        ops.extend(list(s_final_state))

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # sample = np.argmax( s_probsv[0] )
        # sample = np.random.choice( vocab_size, p=np.ravel(s_probsv[0]) )
        temp = temperature_sample(np.ravel(s_probsv[0]), .7)
        sample = temp
        logger.info(temp)

        pred = data_loader.chars[sample]
        ret = ret + ' ' + pred + ' '
        char = pred

    return ret

# ==================================================================

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )
print "FOUND %d BATCHES" % data_loader.num_batches

for j in xrange(200):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in xrange( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%100==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            
            k = 0
            with open("check_poems_temp/test_{}.txt".format(j), 'w') as f:
                while k < 20:
                    prime = data[np.random.randint(0,len(data))]
                    while len(prime) < 1:
                        prime = data[np.random.randint(0,len(data))]
                    try:
                        text = sample(num=100, prime=prime)
                        f.write(text + '\n')
                        k += 1
                    except Exception as e:
                        f.write("UNICODE ERROR: {}\n".format(e))

i = 0
with open("final_temperature.txt", 'w') as f:
    while i < 20:
        prime = data[np.random.randint(0,len(data))]
        while len(prime) < 1:
            prime = data[np.random.randint(0,len(data))]
        try:
            f.write(sample(num=100, prime=prime) + '\n')
            i += 1
        except:
            print("UNICODE ERROR")
            

summary_writer.close()
