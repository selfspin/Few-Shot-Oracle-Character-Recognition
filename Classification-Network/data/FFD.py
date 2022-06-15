import os

import PIL
import numpy as np
from PIL import Image
BPLINE_BOARD_SIZE = 3


def Bspline_Ffd(srcimg: np.ndarray, row_block_num, col_block_num, grid_points):
    dstimg = np.zeros(srcimg.shape)
    delta_x = srcimg.shape[1] * 1.0 / col_block_num
    delta_y = srcimg.shape[0] * 1.0 / row_block_num

    grid_rows = row_block_num + BPLINE_BOARD_SIZE
    grid_cols = col_block_num + BPLINE_BOARD_SIZE

    for y in range(srcimg.shape[0]):
        for x in range(srcimg.shape[1]):
            y_block = y / delta_y
            x_block = x / delta_x
            i = np.floor(y_block)
            j = np.floor(x_block)
            u = x_block - j
            v = y_block - i

            pX = [
                (1 - u * u * u + 3 * u * u - 3 * u) / 6.0,
                (4 + 3 * u * u * u - 6 * u * u) / 6.0,
                (1 - 3 * u * u * u + 3 * u * u + 3 * u) / 6.0,
                u * u * u / 6.0
            ]

            pY = [
                (1 - v * v * v + 3 * v * v - 3 * v) / 6.0,
                (4 + 3 * v * v * v - 6 * v * v) / 6.0,
                (1 - 3 * v * v * v + 3 * v * v + 3 * v) / 6.0,
                v * v * v / 6.0
            ]

            Tx = 0.0
            Ty = 0.0

            for m in range(4):
                for n in range(4):
                    control_point_x = j + n
                    control_point_y = i + m

                    temp = pY[m] * pX[n]
                    Tx += temp * grid_points[0, int(control_point_y * grid_cols + control_point_x)]
                    Ty += temp * grid_points[1, int(control_point_y * grid_cols + control_point_x)]

                    src_x = x + Tx
                    src_y = y + Ty
                    x1 = int(np.floor(src_x))
                    y1 = int(np.floor(src_y))
                    if x1 < 1 or x1 >= srcimg.shape[1] - 1 or y1 < 1 or y1 >= srcimg.shape[0] - 1:
                        dstimg[y, x] = 0
                    else:
                        x2 = x1 + 1
                        y2 = y1 + 1
                        pointa = srcimg[y1, x1]
                        pointb = srcimg[y1, x2]
                        pointc = srcimg[y2, x1]
                        pointd = srcimg[y2, x2]
                        gray = (x2 - src_x) * (y2 - src_y) * pointa - \
                               (x1 - src_x) * (y2 - src_y) * pointb - \
                               (x2 - src_x) * (y1 - src_y) * pointc + \
                               (x1 - src_x) * (y1 - src_y) * pointd
                        dstimg[y, x] = gray
    return dstimg


def init_bpline_para(row_block_num, col_block_num, min_range, max_range, distribution):
    grid_rows = row_block_num + BPLINE_BOARD_SIZE
    grid_cols = col_block_num + BPLINE_BOARD_SIZE
    grid_size = grid_rows * grid_cols
    if distribution == 'uniform':
        return np.random.rand(2, grid_size) * (max_range - min_range) + min_range
    elif distribution == 'gaussian':
        return np.random.randn(2, grid_size) * max_range + min_range


def ffd(img, row_block_num=10,
             col_block_num=10, min_range=-3,
             max_range=3, distribution='uniform'):
    img = np.array(img).reshape(img.size[0], img.size[1])
    grid_points = init_bpline_para(row_block_num, col_block_num, min_range, max_range, distribution)
    tr_img = 255 - Bspline_Ffd(255 - img, row_block_num, col_block_num, grid_points)
    return Image.fromarray(tr_img)


if __name__ == '__main__':
    root = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=1, type='train')
    template = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=1, type='train')
    class_to_oracle = np.load('data/oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
    oracle_to_class = np.load('data/oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()

    files = None
    img = None
    path = None
    for oracle in oracle_to_class.keys():
        path = os.path.join(root, oracle)
        files = os.listdir(path)
        break

    for file in files:
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            img = Image.open(os.path.join(path, file))
            # img = np.array(img.getdata()).reshape(img.size[0], img.size[1])
            break
