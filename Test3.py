# _*_ coding : utf-8 _*_
# @Time : 2023/6/12 15:05
# @Author : 刘炳宁
# @File : Test3
# @Project : Answer

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator#, img_to_array, load_img
from keras.utils import image_utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

#忽略所有与 MatplotlibDeprecationWarning 相关的警告消息
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


#1.加载模型
model = load_model('./model/data/model14_2_VGG 16_cats_vs_dogs_1.h5')
model.summary()
test_datagen = ImageDataGenerator(
    rescale = 1./255
)
test_generator = test_datagen.flow_from_directory(
    directory = "./train_dataset/test2",
    target_size = (224,224),
    shuffle = False,    #不打乱数据集
    class_mode = 'binary',
    batch_size = 20
)
#print(test_datagen)
#print(test_generator)

#2.预测
result = model.predict(test_generator)  #y_pred
result = [(int)((result[i][0] + 0.5)/1.0)for i in range(len(result))]   #转为整数
print(result)
test_label = test_generator.classes #y_test
print(test_label)

#3.分类报告
from sklearn.metrics import classification_report
print("分类报告：",classification_report(test_label,result))
print("混淆矩阵：",confusion_matrix(test_label,result))
print("召回率：",recall_score(test_label,result))


#4.绘制混淆矩阵
plt.figure()
predict = ["cat","dog"]
fact = ["cat","dog"]
classes = list(set(fact))
r1 = confusion_matrix(test_label,result)
plt.figure(figsize = (12,10))
confusion = r1
plt.imshow(confusion,cmap = plt.cm.Blues)
indices = range(len(confusion))
indices2 = range(3)
plt.xticks(indices,classes,rotation = 40,fontsize = 18)
plt.yticks([0.00,1.00],classes,fontsize = 18)
plt.ylim(1.5,-0.5)  #设置y的纵坐标的上下限
plt.title("Confusion matrix",fontdict = {'weight':'normal','size':18})

#5.设置color bar 的标签大小
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 18)
plt.xlabel('Predict label',fontsize = 18)
plt.ylabel('True label',fontsize = 18)
print("len(confusion)",len(confusion))
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        if confusion[first_index][second_index] > 200:
            color = 'black'
        else:
            color = "black"
        plt.text(first_index,second_index,confusion[first_index][second_index],fontsize = 18,color = color,verticalalignment = 'center',horizontalalignment = 'center')
plt.show()

#6.绘画错分报告
#使用迭代器选取图片的x_test-遍历整个test2文件夹-test2中有1000张图片
count = 0
it = iter(test_generator)
x_test,_ = next(test_generator)
print(_)
for x in it:
    yy,_ = x
    x_test = np.concatenate((x_test,yy),axis = 0)
    count += 1
    if count == 100:
        break
#7.绘画错分报告
result = np.reshape(result,(-1,1))
test_label = np.reshape(test_label,(-1,1))
ins = test_label != result
diff_index = np.where(ins == True)[0]   #查找不相同的下标
#print("diff_index:",diff_index)
numForPaint = 8 #只选取前8张错分图片
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']    #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负
lables = ['猫','狗']
for i in range(numForPaint):    #只显示前8个
    j = diff_index[i]
    img = x_test[j]
    y_t = test_label[j][0]
    y_p = result[j][0]
    plt.subplot(2,4,i+1,xticks = [],yticks = [])    #2*4子图显示
    plt.imshow(img) #黑白显示
    plt.title(f'{lables[y_t]} --> {lables[y_p]}')   #显示标题
    plt.subplots_adjust(wspace = 0.1,hspace = 0.2)  #调整子图间距
plt.show()