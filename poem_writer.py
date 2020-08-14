
# coding: utf-8

# In[1]:

#Chinese poem writer#
import time
import numpy as np
import tensorflow as tf


# In[2]:

#read out training poems
with open('libai.txt','r') as f:
    text = f.read()
    
#covert text to a character set, in which characters are different from each other
vocab = set(text)
#create a dictionary, key is character, value is the number
vocab_to_int = {c: i for i,c in enumerate(vocab)}
#create a dictionary, key is the number, value is character
int_to_vocab = dict(enumerate(vocab))

#convert original text to encoded numbers
encoded = np.array([vocab_to_int[c] for c in text],dtype=np.int32)
print('Encoded poetry is: \n',encoded)


# In[3]:

#define a function, which will return the training input and output data
def get_batches(arr, n_seqs, n_steps):
    #get the number of characters per batch
    characters_per_batch = n_seqs * n_steps
    #get the number of batches
    n_batches = len(arr)//characters_per_batch
    #get the data based on the batches
    arr = arr[:n_batches * characters_per_batch]
    #reshape dataset
    arr = arr.reshape((n_seqs,-1))
    
    #return input and output data
    for n in range(0,arr.shape[1],n_steps):
        x = arr[:,n:n+n_steps]
        y = np.zeros_like(x)
        y[:,:-1], y[:,-1] = x[:,1:], x[:,0]
        yield x,y


# In[4]:

#initialize the parameters
def build_inputs(batch_size, num_steps):
    inputs = tf.placeholder(tf.int32,[batch_size,num_steps],name='inputs')
    targets = tf.placeholder(tf.int32,[batch_size,num_steps],name='targets')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    print('input in function build_inputs is: \n', input)
    return inputs, targets, keep_prob


# In[40]:

#build lstm structure
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    def lstm_cell():
        #add basic lstm cell with hidden layers of lstm_size
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse = tf.get_variable_scope().reuse)
        #add dropout of the hidden layers inside the basic lstm cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
        return drop
    #add multiple lstm cells, the number is num_layers
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
    #initialize all states to zero
    initial_state = cell.zero_state(batch_size, tf.float32)
    print('Batch_size: ', batch_size)
    return cell, initial_state


# In[41]:

#create output layer, include softmax
def build_output(lstm_output, in_size, out_size):
    seq_output = tf.concat(lstm_output, axis = 1)
    x = tf.reshape(seq_output,[-1, in_size])
    
    #create variable scope, due to the variables are reused for lstm cells
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size,out_size),stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
        
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name = 'predictions')
    return out, logits


# In[42]:

#create cross entropy loss
def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets,num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss


# In[43]:

#create gradient optimizer
def build_optimizer(loss, learning_rate, grad_clip):
    #get all training variables
    tvars = tf.trainable_variables()
    #use L2 to optimize gradients
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads,tvars))
    return optimizer


# In[44]:

#Create training process
class CharRNN:
    def __init__(self, num_classes, batch_size=4, num_steps=20, lstm_size=5,                  num_layers=5, learning_rate=0.01,grad_clip=5,sampling=False):
        #choose sampling type
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        #reset graph    
        tf.reset_default_graph()
        #get initial parameters
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        #build LSTM struct
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        #get outputs after LSTM
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state = self.initial_state)
        self.final_state = state
        #get prediction data
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


# In[45]:

#define hyper training parameters#
#batch size
batch_size = 4
#number of steps
num_steps = 20
#number of hidden layers in LSTM basic cell
lstm_size = 5
#number of LSTM cells
num_layers = 5
#learning rate
learning_rate = 0.001
#keep prob
keep_prob = 0.5
#running epoches
epochs = 100
#save parameters every n steps
save_every_n = 20


# In[46]:

#build LSTM model
model = CharRNN(len(vocab), batch_size = batch_size, num_steps = num_steps,                 lstm_size = lstm_size, num_layers = num_layers, learning_rate = learning_rate)
saver = tf.train.Saver(max_to_keep = 5)


# In[47]:

#training model#
with tf.Session() as sess:
    #initialize variables
    sess.run(tf.global_variables_initializer())
    
    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x,y in get_batches(encoded,batch_size,num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                   model.targets: y,
                   model.keep_prob: keep_prob,
                   model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,model.final_state,model.optimizer],feed_dict = feed)
            end = time.time()
            print('Epoch: {}/{}...'.format(e+1,epochs),
                 'Training Step: {}...'.format(counter),
                 'Training loss: {:.4f}...'.format(batch_loss),
                 '{:.4f} sec/batch'.format((end-start)))
            if(counter % save_every_n == 0):
                saver.save(sess,"checkpoints/i{}_l{}.ckpt".format(counter,lstm_size))
        saver.save(sess,"checkpoints/i{}_l{}.ckpt".format(counter,lstm_size))


# In[49]:

#predict process#


# In[50]:

#select one item from top n elements
def pick_top_n(preds, vocab_size, top_n = 500):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p/np.sum(p)
    c = np.random.choice(vocab_size,1,p=p)[0]
    return c


# In[54]:

#deifine the prediction information
n_samples, lstm_size, prime = 5, 5, "仙人"

#get sample characters
samples = [c for c in prime]
print(samples)

#initialize predict model
model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
print(model)

saver = tf.train.Saver()
#get last checkpoint
checkpoint = tf.train.latest_checkpoint('checkpoints')

with tf.Session() as sess:
    #restore the checkpoint
    saver.restore(sess, checkpoint)
    #initial state to zero
    new_state = sess.run(model.initial_state)
    for c in prime:
        print(c)
        x = np.zeros((1,1))
        x[0,0] = vocab_to_int[c]
        feed = {model.inputs: x,
               model.keep_prob: 1.,
               model.initial_state: new_state}
        preds, new_state = sess.run([model.prediction,model.final_state],feed_dict = feed)
        print(preds)
        print(np.sum(preds))
    c = pick_top_n(preds,len(vocab))
    samples.append(int_to_vocab[c])
    print(samples)
    for i in range(n_samples):
        x[0,0] = c
        feed = {model.inputs: x,
               model.keep_prob: 1.,
               model.initial_state: new_state}
        preds, new_state = sess.run([model.prediction,model.final_state],
                                   feed_dict = feed)
        c = pick_top_n(preds,len(vocab))
        samples.append(int_to_vocab[c])
        print(samples)
    print(''.join(samples))

