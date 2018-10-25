
# coding: utf-8

# In[1]:


import os

os.environ['CUDA_VISISBLE_DEVICES']='0'

import tensorflow as tf
import tensorlayer as tl



# In[2]:


from ETL import PutImage
from ETL import QueryWorldCoordinate


# In[3]:


import numpy as np
fv=np.zeros((128,128,128),dtype=np.float32)


# In[4]:


print fv.shape


# In[5]:


testname=PutImage(fv)


# In[6]:


print testname


# In[7]:


sess=tf.InteractiveSession()


# In[8]:


with tf.device('/gpu:0'):

        x = tf.placeholder(tf.float32, shape=[None, 32])
        y_ = tf.placeholder(tf.float32, shape=[None, 32])

        re=x+y_
        


# In[14]:


init=tf.global_variables_initializer()
sess.run(init)


# In[28]:


lx=np.array(range(1024),dtype=np.float32)
ly=np.array(range(1024),dtype=np.float32)
lz=np.array(range(1024),dtype=np.float32)
inv=np.eye(4)

print inv


print lx,ly,lz


# In[38]:


for i in range(1024):
    v=QueryWorldCoordinate(testname,lx,ly,lz,inv)

    d=v.reshape(-1,32)

    d2=sess.run(re,feed_dict={x:d,y_:d})
    print i


# In[34]:




