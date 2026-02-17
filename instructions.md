代码运行说明文档

1.original_data存放亚马逊原始数据集，包含Grocery_and_Gourmet_Food、CDs_and_Vinyl、Office_Products这三个。

2.preprocess为数据预处理文件夹，以Grocery_and_Gourmet_Food流程为示意图。

①运行preprocess1.ipynb后在preprocess/dict文件夹中生成以下文件。

![image-20251113221012519](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221012519.png)

②运行preprocess2.ipynb后在preprocess/dict文件夹中生成以下文件。

![image-20251113221035700](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221035700.png)

②运行preprocess3.ipynb后在preprocess文件夹中生成以下文件。

![image-20251113221057492](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221057492.png)

④运行download_pics.py爬取图片，保存在img文件夹中。

![image-20251113221119923](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221119923.png)

⑤运行preprocessImg.py在CODE/datasets文件中生成相应数据集的imgMatrix.npy。

![image-20251113221149355](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221149355.png)

⑥运行preprocessText.py在CODE/datasets文件中生成相应数据集的textMatrix.npy。

![image-20251113221214877](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221214877.png)

⑦运行pca.py后在CODE/datasets文件中生成相应数据集的pca后的图片和文本npy文件。

![image-20251113221235294](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221235294.png)

⑧运行preprocess4.ipynb后在CODE/datasets文件中生成相应数据集的new_train.txt和new_test.txt。

![image-20251113221257742](C:\Users\zyt\AppData\Roaming\Typora\typora-user-images\image-20251113221257742.png)

对其余数据集同样执行上述八个步骤，得到CODE/datasets文件夹下相应数据集的文件，至此数据预处理部分结束。