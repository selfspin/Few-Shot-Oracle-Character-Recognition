## Pre-train Model
You can get the pretrained model on [Google Drive](https://drive.google.com/file/d/1y6-0RqzdqrExDkHC0BXOzIRUEl_Ei1da/view?usp=sharing).

## Preparing the Dataset
我们采用Oracle-source的无标注数据训练，在训练前，需要在`npz_sum.txt`文件中放入训练数据地址，例如

```
../Oracle/data/oracle_source/oracle_source_seq/oracle_source_seq.npz
```

然后运行`generate_dataset.py`文件，获得生成的数据集

若路径报错需创建路径空文件夹

## Training

训练模型

```shell
python main.py train models/SketchTransformer/config/sketch_transformer.yml
```

训练之前您需要在 `models/SketchTransformer/config/sketch_transformer.yml` 中设置好生成的数据集的路径，例如

```
sum_path: 'Oracle/memmap_sum.txt'
offset_path: 'Oracle/offsets.npz'
```

以及预训练模型的路径

```
restore_checkpoint_path: 'checkpoint/pre-train.pth.tar'
```

最终模型在`model_logs`文件夹中

我们训练好的模型在 `checkpoint/best_ckpt.pth.tar` 中

## Generate

先运行甲骨文识别网络文件夹`../Oracle`中的`data/getclass.py`, 其会将甲骨文和数字一一对应，并获得两个`.npy`文件在`data/oracle_fs/img/`文件夹下

运行`generate_oraclefs.py `获得生成的数据集

生成Oracle-FS的数据增强结果

```
python augmentor test models/SketchTransformer/config/generate_oracle_fs.yml
```

运行之前您需要在 `models/SketchTransformer/config/generate_oracle_fs.yml` 中设置好训练完的最终模型地址，例如

```
restore_checkpoint_path: 'checkpoint/best_ckpt.pth.tar'
```

生成结果在 `Generate_oracle_fs/` 文件夹下，请把这整个文件夹放到甲骨文识别网络文件夹中的`data/` 下
