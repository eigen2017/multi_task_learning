import random
import numpy as np

class CsvLoader():
    def csv_to_listoflist(self, filename, type_fn):
        dat = open(filename, 'r')

        listOfList = []
        for oneLine in dat.readlines():
            if oneLine.strip() == '':
                listOfList.append([])
                continue
            listOfList.append([type_fn(oneStr.strip()) for oneStr in oneLine.split(',')])

        dat.close()

        return listOfList

    def get_multi_hot_label(self, class_vocabulary, src_word_list):
        multi_hot_label = [0] * len(class_vocabulary)
        for one_src_word in src_word_list:
            if (one_src_word not in class_vocabulary):
                continue
            multi_hot_label[class_vocabulary.index(one_src_word)] = 1
        return multi_hot_label


class BatchDataLoader():
    def __init__(self, batch_size, dat_file, label_file):
        self._batch_size = batch_size
        csvLoader = CsvLoader()
        self._dat = csvLoader.csv_to_listoflist(dat_file, type_fn=float)
        label = csvLoader.csv_to_listoflist(label_file, type_fn=int)
        self._label = [csvLoader.get_multi_hot_label(range(5), src_word_list) for src_word_list in label]
        self._sample_cnt = min(len(self._dat), len(self._label))
        self._re_shuffle()

    def _re_shuffle(self):
        self._batch_idx = 0
        idxs = range(self._sample_cnt)
        random.shuffle(idxs)
        self._dat = [self._dat[idx] for idx in idxs]
        self._label = [self._label[idx] for idx in idxs]

    def get_a_mini_batch(self):
        from_idx = self._batch_size * self._batch_idx
        to_idx = self._batch_size * (self._batch_idx + 1)
        if (to_idx > self._sample_cnt):
            self._re_shuffle()
            return [], []
        self._batch_idx = self._batch_idx + 1

        data = self._dat[from_idx:to_idx]
        label = self._label[from_idx:to_idx]

        return data, label


if __name__ == "__main__":
    trainingDataLoader = BatchDataLoader(256, '../data_with_shrinked_label/train_dat.csv', '../data_with_shrinked_label/train_label.csv')
    idx = 0
    for _ in range(1):
        while (True):
            data, label = trainingDataLoader.get_a_mini_batch()
            if (data == []):
                break

            print idx, len(data), len(label)
            print data[0]
            print label[0]

            idx = idx + 1
