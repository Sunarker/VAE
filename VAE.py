import tensorflow as tf
import numpy as np
import utils as U
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from tensorflow.examples.tutorials.mnist import input_data

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

slim = tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
Normal = tf.contrib.distributions.Normal

# ---------------------------------------------------------------
#### hyparameters
D = 200			# dimension of the latent variable

#### graph definition
sample_n = tf.placeholder(tf.int32, name = 'repeating_times_of_each_sample')
x = tf.placeholder(tf.float32, shape=(None,784), name='x')
x_ = tf.tile(x, [sample_n, 1])

net = x_
net = slim.stack(net,slim.fully_connected,[D])
gaussian_params = slim.fully_connected(net, D*2, activation_fn=None)
q_z = Normal(gaussian_params[:,:D], 1e-6 + tf.nn.softplus(gaussian_params[:,D:]))
z = tf.to_float(q_z.sample())
net = slim.stack(z,slim.fully_connected,[D])
logits_x = slim.fully_connected(net,784,activation_fn=None)
p_x = Bernoulli(logits=logits_x)

p_z = Normal(tf.zeros_like(gaussian_params[:,:D]),tf.ones_like(gaussian_params[:,D:]))

# loss
logP = tf.reduce_sum(p_x.log_prob(x_),1) - tf.reduce_sum(q_z.log_prob(z),1) + tf.reduce_sum(p_z.log_prob(z),1) 
logF = tf.transpose(tf.reshape(logP,[sample_n,-1]))
iwae1 = U.logSumExp(logF, axis=1) - tf.log(tf.to_float(sample_n))
iwae2 = -U.logSumExp(-logF, axis=1) + tf.log(tf.to_float(sample_n))

loss_vae = -tf.reduce_mean(logP)
loss_iwae1 = -tf.reduce_mean(iwae1)
loss_iwae2 = -tf.reduce_mean(iwae2)

# evaluation
loglikelihood = tf.reduce_mean(iwae1)

#### training optimizer
train_op_vae=tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss_vae)
train_op_iwae1=tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss_iwae1)
train_op_iwae2=tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss_iwae2)

# ---------------------------------------------------------------
#### black-on-white MNIST (harder to learn than white-on-black MNIST)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#### training and evaluation
batch_size1 = 24
batch_size2 = 5
iter_num = 100000

data_vae1 = []
data_vae2 = []
with tf.train.MonitoredSession() as sess:
  for i in range(1,iter_num):
    batch = mnist.train.next_batch(batch_size1)
    res = sess.run([train_op_vae, loss_vae], {x: batch[0], sample_n: 1})
    if i % 1000 == 1:
      data_vae1.append([i] + res[1:])
      print('Step %d, Training Loss: %0.3f' % (i,res[1]))
   
    if i % 1000 == 1:
      avg_res = 0
      for j in range(2000):
        batch2 = mnist.test.next_batch(batch_size2)
        res2 = sess.run(loglikelihood,{x: batch2[0], sample_n: 100})
        avg_res += res2
      avg_res /= 2000
      data_vae2.append([i,avg_res])
      print('Step %d, Test Loglikelihood: %0.3f' % (i,avg_res))
    
data_iwae11 = []
data_iwae12 = []
with tf.train.MonitoredSession() as sess:
  for i in range(1,iter_num):
    batch = mnist.train.next_batch(batch_size1)
    res = sess.run([train_op_iwae1, loss_iwae1], {x: batch[0], sample_n: 20})
    if i % 1000 == 1:
      data_iwae11.append([i] + res[1:])
      print('Step %d, Training Loss: %0.3f' % (i,res[1]))
   
    if i % 1000 == 1:
      avg_res = 0
      for j in range(2000):
        batch2 = mnist.test.next_batch(batch_size2)
        res2 = sess.run(loglikelihood,{x: batch2[0], sample_n: 100})
        avg_res += res2
      avg_res /= 2000
      data_iwae12.append([i,avg_res])
      print('Step %d, Test Loglikelihood: %0.3f' % (i,avg_res))
    
data_iwae21 = []
data_iwae22 = []
with tf.train.MonitoredSession() as sess:
  for i in range(1,iter_num):
    batch = mnist.train.next_batch(batch_size1)
    res = sess.run([train_op_iwae2, loss_iwae2], {x: batch[0], sample_n: 20})
    if i % 1000 == 1:
      data_iwae21.append([i] + res[1:])
      print('Step %d, Training Loss: %0.3f' % (i,res[1]))
   
    if i % 1000 == 1:
      avg_res = 0
      for j in range(2000):
        batch2 = mnist.test.next_batch(batch_size2)
        res2 = sess.run(loglikelihood,{x: batch2[0], sample_n: 100})
        avg_res += res2
      avg_res /= 2000
      data_iwae22.append([i,avg_res])
      print('Step %d, Test Loglikelihood: %0.3f' % (i,avg_res))
    

#### plot
data_vae1 = np.array(data_vae1).T
data_vae2 = np.array(data_vae2).T
data_iwae11 = np.array(data_iwae11).T
data_iwae12 = np.array(data_iwae12).T
data_iwae21 = np.array(data_iwae21).T
data_iwae22 = np.array(data_iwae22).T

f,axarr=plt.subplots(1,2,figsize=(18,6))

axarr[0].plot(data_vae1[0],data_vae1[1],'-',linewidth=4.0)
axarr[0].plot(data_iwae11[0],data_iwae11[1],'--',linewidth=4.0)
axarr[0].plot(data_iwae21[0],data_iwae21[1],'-.',linewidth=4.0)
axarr[0].set_title('Training Loss')
axarr[0].grid(True)
axarr[0].legend(['vae','iwae','iwae_t'],loc='upper left')

axarr[1].plot(data_vae2[0],data_vae2[1],'-',linewidth=4.0)
axarr[1].plot(data_iwae12[0],data_iwae12[1],'--',linewidth=4.0)
axarr[1].plot(data_iwae22[0],data_iwae22[1],'-.',linewidth=4.0)
axarr[1].set_title('Loglikelihood')
axarr[1].grid(True)
axarr[1].legend(['vae','iwae','iwae_t'],loc='upper left')

f.savefig('VAE.png')

with open('VAE.pkl','w') as w:
   pickle.dump({'data_vae':[data_vae1,data_vae2], 'data_iwae1':[data_iwae11,data_iwae12], 'data_iwae2': [data_iwae21,data_iwae22]},w)
