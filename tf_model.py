import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import get_dummies

data = pd.read_csv(r'C:\Users\deepa\Downloads\Iris.csv', index_col = 0)

cols = data.columns
features = cols[0:4]
labels = cols[4]
print(features)
print(labels)


data_norm = pd.DataFrame(data)

for feature in features:
    data[feature] = (data[feature] - data[feature].mean())/data[feature].std()

#Show that should now have zero mean
print("Averages")
print(data.mean())

print("\n Deviations")
#Show that we have equal variance
print(pow(data.std(),2))


#Shuffle The data
indices = data_norm.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)
X = data_norm.reindex(indices)[features]
y = data_norm.reindex(indices)[labels]

# One Hot Encode as a dataframe
y = get_dummies(y)

# Generate Training and Validation Sets
train_pct_index = int(0.8 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]


# Convert to np arrays so that we can use with TensorFlow
X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test).astype(np.float32)



#Check to make sure split still has 4 features and 3 labels
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Modelling
training_size = X_train.shape[1]
test_size = X_test.shape[1]
num_features = 4
num_labels = 3

num_hidden = 10

graph = tf.Graph()
with graph.as_default():
    tf.disable_v2_behavior()
    tf_train_set = tf.constant(X_train)
    tf_train_labels = tf.constant(y_train)
    tf_valid_set = tf.constant(X_test)

    print(tf_train_set)
    print(tf_train_labels)

    ## Note, since there is only 1 layer there are actually no hidden layers
    ## there would be num_hidden
    weights_1 = tf.Variable(tf.random.truncated_normal([num_features, num_hidden]))
    weights_2 = tf.Variable(tf.random.truncated_normal([num_hidden, num_labels]))
    ## tf.zeros Automaticaly adjusts rows to input data batch size
    bias_1 = tf.Variable(tf.zeros([num_hidden]))
    bias_2 = tf.Variable(tf.zeros([num_labels]))

    logits_1 = tf.matmul(tf_train_set, weights_1) + bias_1
    rel_1 = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(rel_1, weights_2) + bias_2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(.005).minimize(loss)


    ## Training prediction
    predict_train = tf.nn.softmax(logits_2)


    # Validation prediction
    logits_1_val = tf.matmul(tf_valid_set, weights_1) + bias_1
    rel_1_val = tf.nn.relu(logits_1_val)
    logits_2_val = tf.matmul(rel_1_val, weights_2) + bias_2
    predict_valid = tf.nn.softmax(logits_2_val)



def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


num_steps = 10000
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(loss.eval())
    for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, predict_train])

        if (step % 2000 == 0):
            # print(predictions[3:6])
            print('Dagrunner Loss at step %d: %f' % (step, l))
            print('Dagruner Training accuracy: %.1f%%' % accuracy(predictions, y_train[:, :]))
            print('Dagrunner Validation accuracy: %.1f%%' % accuracy(predict_valid.eval(), y_test))
            np.savetxt(r'C:\Users\deepa\Downloads\out2.csv', predict_valid.eval(), delimiter=",")
            np.savetxt(r'C:\Users\deepa\Downloads\out3.csv', y_test, delimiter=",")


Predicted_Value = pd.read_csv(r'C:\Users\deepa\Downloads\out2.csv', header=None)
Predicted_Value = Predicted_Value.round(decimals = 0)
Predicted_Value.columns = y.columns.values
Predicted_Value = pd.DataFrame(Predicted_Value.unstack())
Predicted_Value = Predicted_Value.loc[(Predicted_Value != 0).any(axis=1)]
Predicted_Value = Predicted_Value.reset_index(level=[0,1])
Predicted_Value = Predicted_Value.sort_values('level_1')
Predicted_Value = Predicted_Value.drop(0, axis=1)
Predicted_Value.rename(columns = {'level_0':'Predicted_Species'}, inplace = True)
Predicted_Value = Predicted_Value.set_index('level_1')
Predicted_Value.to_csv(r'C:\Users\deepa\Downloads\Predicted_Value.csv')


# Restoring Original test values to compare accuracy
original_y = pd.DataFrame(y_test)
original_y.columns = y.columns.values
original_y = pd.DataFrame(original_y.unstack())
original_y = original_y.loc[(original_y != 0).any(axis=1)]
original_y = original_y.reset_index(level=[0,1])
original_y = original_y.drop(0, axis=1)
original_y = original_y.sort_values('level_1')
original_y = original_y.set_index('level_1')
original_y.rename(columns = {'level_0':'Species'}, inplace = True)


original_x = pd.DataFrame(X_test)
original_x.columns = features

original_x = original_x.join(original_y)


BQ_Dataset = original_x.join(Predicted_Value)

BQ_Dataset.to_csv(r'C:\Users\deepa\Downloads\BQ_Dataset.csv')



