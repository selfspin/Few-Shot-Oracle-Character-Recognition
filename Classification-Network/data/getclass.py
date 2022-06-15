import os
import numpy as np


def get_class(path):
    dbtype_list = os.listdir(path)
    for dbtype in dbtype_list[::]:
        if os.path.isfile(os.path.join(path, dbtype)):
            dbtype_list.remove(dbtype)

    class_to_oracle = {}
    oracle_to_class = {}
    for i in range(len(dbtype_list)):
        oracle_to_class[dbtype_list[i]] = i
        class_to_oracle[i] = dbtype_list[i]

    np.save('oracle_fs/img/class_to_oracle.npy', class_to_oracle)
    np.save('oracle_fs/img/oracle_to_class.npy', oracle_to_class)
    return


if __name__ == '__main__':
    get_class('oracle_fs/img/oracle_200_1_shot/train')
