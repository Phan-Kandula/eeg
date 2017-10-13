import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from pandas import DataFrame
import tensorflow as tf

#dataprocessing
dataframe = pd.read_csv("data.csv")
inputX = dataframe.iloc[:, 1:].values
inputY = dataframe.iloc[:, 0]
LabelEncoderY = LabelEncoder()
inputY = LabelEncoderY.fit_transform(inputY)
inputY = np.reshape(inputY, (inputY.size, 1))
ohenc = OneHotEncoder(categorical_features="all")
inputY = ohenc.fit_transform(inputY).toarray()

x_train, x_test, y_train, y_test : train_test_split(inputX, inputY, test_size = 0.33, random_state = 81)

# setting up neural network
nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500
n_classes = 4
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hl1 = {'weights': tf.Variable(tf.random_normal([len(x_train),nodes_hl1])),
        'biases': tf.Variable(tf.random_normal([nodes_hl1]))}
hl2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1,nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([nodes_hl2]))}
hl3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2,nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([nodes_hl3]))}
output_layer: {'weights' : tf.Variable(tf.random_normal([nodes_hl3,n_classes])),
                'biases' : tf.Variable(tf.random_normal([n_classes]))}
def neural_network_model(data):
    
    # layer = (data*weights) + biases 
    l1 = tf.add(tf.multiply(data,hl1['weights']), hl1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.multiply(data,hl2['weights']), hl2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.multiply(data,hl3['weights']), hl3['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.multiply(data,output_layer['weights']), output_layer['biases'])
    
    return output

def train_set(x)
    prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size
				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

	    
train_neural_network(x)