### 操作步骤
1. 在 ```Researh/tableGAN``` 目录下运行  ```pip3 install -r requirements.txt```
2. 在 ```Researh/tableGAN``` 目录下运行 
   ```python3 main.py --train --dataset=Adult --epoch=5000 --test_id=OI_11_00``` 开始训练 Rule Model。
3. 每训练 50 轮 会在 ```Research/tableGAN/samples/Adult/Adult_rm_pred.csv``` 路径下输出 Rule Model 的输出。

### 关键代码内容
#### 1. 将数据转化为 CNN 的输入
```922 load_dataset()```

#### 2. Table GAN
```Discriminator: 684```
```Generator: 804```
```loss func: 311```
#### 3. Rule Model
```model: 1272```
```loss func: 1163```
