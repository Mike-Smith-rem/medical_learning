'''Copyright oyk
Created 07 09:07:32
'''
import numpy as np


def iid(dataset, num_clients, num_class=3) -> dict:
    """
        Sample I.I.D. client data from dataset
        preview: the dataset should have classes one by one,
                if use binary segment, the best is set num_class as 1
        :param dataset:
        :param num_clients:
        :return: dict of image index
    """
    class0_items = int(len(dataset) / num_clients / num_class)
    dict_clients, class0_all_idxs = {}, [i for i in range(len(dataset) // num_class)]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(class0_all_idxs, class0_items, replace=False))
        class0_all_idxs = list(set(class0_all_idxs) - dict_clients[i])
    for i in range(num_clients):
        class0_index = list(dict_clients[i])
        class_all_index = list(dict_clients[i])
        for j in class0_index:
            t = num_class
            for k in range(1, t):
                class_all_index.append(j + len(dataset) // num_class * k)
        dict_clients[i] = set(class_all_index)

    # print the answer
    # for i in range(len(dict_clients)):
    #     d = dict_clients[i]
    #     print(sorted(d))
    return dict_clients
