from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np

size = (224,224)
ba = 128

trainD = 'Rice_Image_Dataset/Split/train'
testD = 'Rice_Image_Dataset/Split/test'
validation = 'Rice_Image_Dataset/Split/val'

trainSet = ImageDataGenerator(preprocessing_function=preprocess_input,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip=True,
                              validation_split = 0.4)

trainDataSet = trainSet.flow_from_directory(trainD,
                                             target_size=(250,250),
                                             batch_size= ba,
                                             class_mode = 'categorical',
                                             subset='training')

valDataSet = trainSet.flow_from_directory(validation,
                                             target_size=(250,250),
                                             batch_size= ba,
                                             class_mode = 'categorical',
                                             subset='training')

testDataSet = trainSet.flow_from_directory(testD,
                                             target_size=(250,250),
                                             batch_size= 1,
                                             class_mode = 'categorical',
                                             subset='training')


base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)

predictions = Dense(trainDataSet.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(trainDataSet,epochs=10)
print(model.evaluate(testDataSet,verbose=2))
model.save('')


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape = (250,250,3)),
#     tf.keras.layers.MaxPool2D(2,2),
#     tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
#     tf.keras.layers.MaxPool2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#     tf.keras.layers.MaxPool2D(2,2)
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512,activation='relu'),
#     tf.keras.layers.Dense(1,activation='sigmoid')
# ]

# )

# model.compile(loss='binary_crossentropy',
#               optimizer = RMSprop(lr=0.001),
#               metrics =['accuracy'])

# model_fit = model.fit(trainDataSet, 
#                       step_per_epoch =3,
#                       epochs = 10)