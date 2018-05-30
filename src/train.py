import tensorflow as tf

from model import MultiTaskFcModel
from read_data import BatchDataLoader

my_model = MultiTaskFcModel()

writer = tf.summary.FileWriter('./log')
writer.add_graph(tf.get_default_graph())
# exit()
trainingDataLoader = BatchDataLoader(1024, '../data_with_shrinked_label/train_dat.csv', '../data_with_shrinked_label/train_label.csv')
devDataLoader = BatchDataLoader(4000, '../data_with_shrinked_label/dev_dat.csv', '../data_with_shrinked_label/dev_label.csv')
traingAccDataLoader = BatchDataLoader(120000, '../data_with_shrinked_label/train_dat.csv', '../data_with_shrinked_label/train_label.csv')
dev_data, dev_label = devDataLoader.get_a_mini_batch()
train_acc_data, train_acc_label = traingAccDataLoader.get_a_mini_batch()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
for one_iteration in range(10000):

    while (True):
        data, label = trainingDataLoader.get_a_mini_batch()
        if (data == []):
            break
        sess.run([my_model.my_optimize_op], feed_dict={my_model.x: data, my_model.y: label, my_model.phase_train: True})

    dev_precision_rate, dev_recall_rate, dev_f1_score = sess.run([my_model.precision_rate, my_model.recall_rate, my_model.f1_score], feed_dict={my_model.x: train_acc_data, my_model.y: train_acc_label, my_model.phase_train: False})
    train_precision_rate, train_recall_rate, train_f1_score = sess.run([my_model.precision_rate, my_model.recall_rate, my_model.f1_score], feed_dict={my_model.x: dev_data, my_model.y: dev_label, my_model.phase_train: False})

    print('epoch:[' + str(one_iteration) + ']')
    print('precision_rate:')
    print(train_precision_rate)
    print(dev_precision_rate)
    print('recall_rate:')
    print(train_recall_rate)
    print(dev_recall_rate)
    print('f1_score:')
    print(train_f1_score)
    print(dev_f1_score)

sess.close()
