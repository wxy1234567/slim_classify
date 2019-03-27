Python3使用TF-Slim进行图像分类

# 机器环境
- ubuntu
- python3.6
- tensorflow==1.6.0

# 准备图片数据
- 准备好自定义的图片数据
- 放到 data_prepare/pic/train 和 data_prepare/pic/validation 中
- 自己建立分类文件夹，文件夹名为分类标签名


# 将图片数据转换成TF-Record格式文件
- 在 data_prepare/ 下，执行

```python
python data_convert.py -t pic/ \
  --train-shards 2 \
  --validation-shards 2 \
  --num-threads 2 \
  --dataset-name satellite
```
- 会生成4个tf-record文件和1个label文件

下载mobilnet模型到slim/model下

# 在 slim/ 文件夹下执行如下命令，进行训练：

```python
python train_image_classifier.py \
  --train_dir=flowers/train_log \
  --dataset_name=flowers \
  --train_image_size=299 \
  --dataset_split_name=train \
  --dataset_dir=data \
  --model_name="mobilenet_v2_140" \
  --checkpoint_path=model/mobilenet_v2_1.4_224.ckpt \
  --checkpoint_exclude_scopes=MobilenetV2/Logits,MobilenetV2/AuxLogits \
  --trainable_scopes=MobilenetV2/Logits,MobilenetV2/AuxLogits \
  --max_number_of_steps=20000 \
  --batch_size=16 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --label_smoothing=0.1 \
  --num_clones=1 \
  --num_epochs_per_decay=2.5 \
  --moving_average_decay=0.9999 \
  --learning_rate_decay_factor=0.98 \
  --preprocessing_name="inception_v2"
```

# 在 slim/ 文件夹下执行如下命令，进行模型能力评估：

```python
python eval_image_classifier.py \
  --checkpoint_path=flowers/train_log \
  --eval_dir=flowers/eval_log \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=data \
  --model_name="mobilenet_v2_140" \
  --batch_size=32 \
  --num_preprocessing_threads=2 \
  --eval_image_size=299
```

# 导出训练好的模型
- 在 slim/ 文件夹下面执行如下命令：

```python
python export_inference_graph.py \
  --alsologtostderr \
  --model_name="mobilenet_v2_140" \
  --image_size=299 \
  --output_file=flowers/export/mobilenet_v2_140_inf_graph.pb \
  --dataset_name flowers
```

- 在 项目根目录 执行如下命令（需将5271改成train_dir中保存的实际的模型训练步数）

```python
python freeze_graph.py \
  --input_graph slim/flowers/export/mobilenet_v2_140_inf_graph.pb \
  --input_checkpoint slim/flowers/train_log/model.ckpt-20000 \
  --input_binary true \
  --output_node_names MobilenetV2/Predictions/Reshape_1 \
  --output_graph slim/flowers/export/frozen_graph.pb
```

# 对单张图片进行预测
- 在 项目根目录 执行如下命令

```python
python classify_image_test.py \
  --model_path slim/flowers/export/frozen_graph.pb \
  --label_path data_prepare/pic/label.txt \
  --image_file test_image.jpg
```
