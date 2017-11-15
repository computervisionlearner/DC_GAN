#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 11:24:13 2017

@author: no1
"""
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data/')

def input_placeholder(img_size,noise_size):
    img=tf.placeholder(dtype=tf.float32,shape=(None,img_size[1],img_size[2],img_size[3]),name='input_image')
    noise=tf.placeholder(dtype=tf.float32,shape=(None,noise_size),name='input_noise')
    return img,noise

def generator(noise,output_dim,is_train=True,alpha=0.01):
    #output_dim =1,means one channel
    with tf.variable_scope('generator') as scope0:
        if not is_train:
            scope0.reuse_variables()
        
        layer1=tf.layers.dense(noise,4*4*512)
        layer1=tf.reshape(layer1,[-1,4,4,512])       
        #batch normalization
        layer1=tf.layers.batch_normalization(layer1,training=is_train)
        layer1=tf.maximum(alpha*layer1,layer1)
        layer1=tf.nn.dropout(layer1,keep_prob=0.8)
        
        #deconv
        layer2=tf.layers.conv2d_transpose(layer1,256,4,strides=1,padding='valid')
        layer2=tf.layers.batch_normalization(layer2,training=is_train)
        layer2=tf.maximum(alpha*layer2,layer2)
        layer2=tf.layers.dropout(layer2,rate=0.8,training=is_train)
        
        layer3=tf.layers.conv2d_transpose(layer2,128,3,strides=2,padding='same')
        layer3=tf.layers.batch_normalization(layer3,training=is_train)
        layer3=tf.maximum(alpha*layer3,layer3)
        layer3=tf.nn.dropout(layer1,keep_prob=0.8)
        
        logits=tf.layers.conv2d_transpose(layer3,output_dim,3,strides=2,padding='same')
        outputs=tf.tanh(logits)
        tf.summary.image('input',outputs,10)
        
        return outputs
    
def discriminator(img_or_noise,reuse=False,alpha=0.01):
    #img_or_noise means batch image placeholder or noise's output
    with tf.variable_scope('discriminator') as scope1:
        
        if reuse:
            scope1.reuse_variables()
        #none*28*28*1 to  none*14*14*128
        layer1=tf.layers.conv2d(img_or_noise,128,3,strides=2,padding='same')
        layer1=tf.maximum(alpha*layer1,layer1)
        layer1=tf.nn.dropout(layer1,keep_prob=0.8)
        
        #none*14*14*128  to   none*7*7*256
        layer2=tf.layers.conv2d(layer1,256,3,strides=2,padding='same')
        layer2=tf.layers.batch_normalization(layer2,training=True)
        layer2=tf.maximum(alpha*layer2,layer2)
        layer2=tf.nn.dropout(layer2,keep_prob=0.8)

        #none*7*7*256   to     4*4*512
        layer3=tf.layers.conv2d(layer2,512,3,strides=1,padding='valid')
        layer3=tf.layers.batch_normalization(layer3,training=True)
        layer3=tf.maximum(alpha*layer3,layer3)
        layer3=tf.nn.dropout(layer3,keep_prob=0.8)
        
        flatten=tf.reshape(layer3,(-1,4*4*512))
        logits=tf.layers.dense(flatten,1)
        outputs=tf.sigmoid(logits)
        return logits,outputs
    
def inference(real_img,fake_noise,image_depth=1,smooth=0.1):
    g_outputs=generator(fake_noise,image_depth,is_train=True)
    d_logits_real,d_outputs_real=discriminator(real_img)
    d_logits_fake,d_outputs_fake=discriminator(g_outputs,reuse=True)
    
    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                        labels=tf.ones_like(d_outputs_fake)*(1-smooth)))
    
    d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                        labels=tf.ones_like(d_outputs_real)*(1-smooth)))
    
    d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                        labels=tf.zeros_like(d_outputs_fake)))
    
    d_loss=tf.add(d_loss_real,d_loss_fake)
    
    tf.summary.scalar('d_loss_real', d_loss_real)
    tf.summary.scalar('d_loss_fake', d_loss_fake)

    
    
    return g_loss,d_loss

def test(fake_placeholder,output_dim=1,num_images=25):
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint('checkpoints'))
        fake_shape=fake_placeholder.shape.as_list()[-1]
        
        fake_images=np.random.uniform(-1,1,size=[num_images,fake_shape])
        
        samples=sess.run(generator(fake_placeholder,output_dim=1,is_train=False),
                         feed_dict={fake_placeholder:fake_images})
        
        result=np.squeeze(samples,-1)
        
        
        plot_image(result)
        
def plot_image(samples):
    
    fig,axes=plt.subplots(nrows=5,ncols=5,figsize=(7,7))
    
    for img,ax in zip(samples,axes.flatten()):        
        ax.imshow(img.reshape((28,28)),cmap='Greys_r')        
        ax.axis('off')
      

def get_optimizer(g_loss,d_loss,beta=0.4,learning_rate=0.001):
#    train_vars=tf.trainable_variables()
#    g_vars=[var for var in train_vars if var.name.startswith('generator')]
#    d_vars=[var for var in train_vars if var.name.startswith('discriminator')]
    g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(g_loss,var_list=g_vars)
        d_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(d_loss,var_list=d_vars)
    
        return g_opt,d_opt
#%%
def train(real_placeholder,fake_placeholder,g_train_opt,d_train_opt,epoches,noise_size=100,data_shape=[-1,28,28,1],batch_size=64,n_samples=25):
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('log/', sess.graph)
        for e in range(1,epoches):
            
            for step in range(mnist.train.num_examples//batch_size):
                images,labels=mnist.train.next_batch(batch_size)
                batch_image=images.reshape(data_shape)
                
                batch_image=batch_image*2 -1
                
                batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_size))
                
                sess.run(g_train_opt,feed_dict={real_placeholder:batch_image,fake_placeholder:batch_noise})
                
                sess.run(d_train_opt,feed_dict={real_placeholder:batch_image,fake_placeholder:batch_noise})
                
                summary_str=sess.run(summary,feed_dict={real_placeholder:batch_image,fake_placeholder:batch_noise})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                
                if step %101 ==0:
                    
                    train_loss_d=d_loss.eval({real_placeholder:batch_image,fake_placeholder:batch_noise})
                    
                    train_loss_g=g_loss.eval(feed_dict={fake_placeholder:batch_noise})
                    
                    print('step:{}/Epoch:{}/total Epoch:{}'.format(step,e,epoches),
                          'Discriminator Loss:{:.4f}..'.format(train_loss_d),'Generator Loss:{:.4f}..'.format(train_loss_g))
    
        saver.save(sess,'./checkpoints/generator.ckpt')
        
# %%       
with tf.Graph().as_default():

    real_img,fake_img=input_placeholder([-1,28,28,1],noise_size=100)
    
    g_loss,d_loss=inference(real_img,fake_img)
    summary = tf.summary.merge_all()
    g_train_opt,d_train_opt=get_optimizer(g_loss,d_loss)
    
    saver=tf.train.Saver()
    train(real_img,fake_img,g_train_opt,d_train_opt,epoches=4)
    test(fake_img,num_images=25)
       
        
#%%        
