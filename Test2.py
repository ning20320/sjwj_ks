# _*_ coding : utf-8 _*_
# @Time : 2023/6/12 8:47
# @Author : 刘炳宁
# @File : Test2
# @Project : Answer

#基于VGG16构建模型
from keras import models,layers
from keras.applications import VGG16
from tensorflow import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#1.训练样本的目录
train_dir = './train_dataset/train/train'

#2.验证样本的目录
validation_dir = './train_dataset/train/validation'

#3.测试样本的目录
test_dir = './train_dataset/train/test'

#4.训练集生成器--训练集数据加强
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'

)
train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (224,224),
    class_mode = 'binary',
    batch_size = 20 #每次扔进神经网络训练的数据为20个
)

#5.验证样本生成器
validation_datagen = ImageDataGenerator(
    rescale = 1./255
)
validation_generator = validation_datagen.flow_from_directory(
    directory = validation_dir,
    target_size = (224,224),
    class_mode = 'binary',
    batch_size = 20
)

#6.测试样本生成器
test_datagen = ImageDataGenerator(
    rescale = 1./255
)
test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (224,224),
    class_mode = 'binary',
    batch_size = 20
)
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

#7.VGG16 实例化-使用imagenet数据集训练，不包含顶层（即全连接层）
conv_base = VGG16(
    weights = 'imagenet',
    include_top = False,    #是否指定模型最后是否包含密集连接分类器
    input_shape = (224,224,3)
)

#8.冻结卷积基-保证其权重在训练过程中不变-不训练这个，参数过多
conv_base.trainable = False

#9.构建网络模型-基于VGG16建立模型
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten(input_shape = conv_base.output_shape[1:])) #图片输出四维，1代表数量
model.add(layers.Dense(256,activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation = 'sigmoid'))   #二分类

#10.定义优化器、代价函数、训练过程中计算准确率
model.compile(
    optimizer = optimizers.Adam(lr = 0.0005/10),
    loss = 'binary_crossentropy',
    metrics = ['acc']
)
model.summary()

#11.拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100, #100, 2000/batch_size
    epochs = 20,   #20,把整个数据集丢进神经网络训练20次
    validation_data = validation_generator,
    validation_steps = 50 #50   1000/batch_size
)

#12.保存模型
model.save('./model/data/model14_2_VGG 16_cats_vs_dogs_1.h5')

#13.评估测试集的准确率
test_eval = model.evaluate_generator(test_generator)
#test_eval = model.evalue(test_generator)
print("测试集准确率：",test_eval)
train_eval = model.evaluate_generator(train_generator)
#train_eval = model.evale(train_generator)
print("训练集准确率：",train_eval)
val_eval = model.evaluate_generator(validation_generator)
#val_eval = model.evalue(validation_generator)
print("验证集准确率：",val_eval)

#14.绘制训练过程中的损失曲线和精度曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc) + 1)
plt.plot(epochs,acc,'bo')
plt.plot(epochs,acc,'b',label = 'Training acc')
plt.plot(epochs,val_acc,'ro')
plt.plot(epochs,val_acc,'r',label = 'Validation acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs,loss,'bo')
plt.plot(epochs,loss,'b',label = 'Training Loss')
plt.plot(epochs,val_loss,'ro')
plt.plot(epochs,val_loss,'r',label = 'Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss and Validation Loss")
plt.legend()
plt.show()
