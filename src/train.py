import tensorflow as tf

from model import MultiTaskFcModel
from read_data import TrainingDataLoader

my_model = MultiTaskFcModel()

writer = tf.summary.FileWriter('./log')
writer.add_graph(tf.get_default_graph())
# exit()
trainingDataLoader = TrainingDataLoader(1024)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for one_iteration in range(10000):

    while (True):
        data, label = trainingDataLoader.get_a_mini_batch()
        if (data == []):
            break
        _, precision_rate, recall_rate = sess.run([my_model.my_optimize_op, my_model.precision_rate, my_model.recall_rate], feed_dict={my_model.x: data, my_model.y: label, my_model.phase_train: True})

    print('epoch:[' + str(one_iteration) + ']')
    print('precision_rate:')
    print(precision_rate)
    print('recall_rate:')
    print(recall_rate)




sess.close()
