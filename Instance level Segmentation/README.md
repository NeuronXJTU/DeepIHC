## Yolact-keras实例分割模型在pytorch当中的实现

## 所需环境
pytorch==1.2.0  
torchvision==0.4.0


## 训练步骤
### a、训练shapes形状数据集
1. 数据集的准备   
将图片(jpg)和对应的json文件放入根目录下的datasets/before文件夹。

2. 数据集的处理   
打开coco_annotation.py，里面的参数默认用于处理shapes形状数据集，直接运行可以在datasets/coco文件夹里生成图片文件和标签文件，并且完成了训练集和测试集的划分。

3. 开始网络训练   
train.py的默认参数用于训练shapes数据集，默认指向了根目录下的数据集文件夹，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolact.py和predict.py。
首先需要去yolact.py里面修改model_path以及classes_path，这两个参数必须要修改。    
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**    
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。



### c、训练coco数据集
1. 数据集的准备  
coco训练集 http://images.cocodataset.org/zips/train2017.zip   
coco验证集 http://images.cocodataset.org/zips/val2017.zip   
coco训练集和验证集的标签 http://images.cocodataset.org/annotations/annotations_trainval2017.zip   

2. 开始网络训练  
解压训练集、验证集及其标签后。打开train.py文件，修改其中的classes_path指向model_data/coco_classes.txt。   
修改train_image_path为训练图片的路径，train_annotation_path为训练图片的标签文件，val_image_path为验证图片的路径，val_annotation_path为验证图片的标签文件。   

3. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolact.py和predict.py。
首先需要去yolact.py里面修改model_path以及classes_path，这两个参数必须要修改。    
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**    
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权值，放入model_data，运行predict.py，输入
### b、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在yolact.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。


### b、评估coco的数据集
1. 下载好coco数据集。  
2. 在yolact.py里面修改model_path以及classes_path。**model_path指向coco数据集的权重，在logs文件夹里。classes_path指向model_data/coco_classes.txt。**    
3. 前往eval.py设置classes_path，指向model_data/coco_classes.txt。修改Image_dir为评估图片的路径，Json_path为评估图片的标签文件。 运行eval.py即可获得评估结果。  
  
## Reference
https://github.com/feiyuhuahuo/Yolact_minimal   
