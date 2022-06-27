import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist                                         #Load data from fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    #Load data from fashion MNIST

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle boot']                                          #Save lable names

train_images = train_images / 255.0
test_images = test_images / 255.0       #Adjust value to 0~1

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),       #Layer 1: 2차원 배열 -> 1차원배열
    tf.keras.layers.Dense(128, activation='relu'),      #Layer 2
    tf.keras.layers.Dense(10)                           #Layer 3
])

model.compile(optimizer='adam',                                                         #Optimizer: 모델이 인식하는 데이터와 손실함수를 기반으로 모델이 업데이트 되는 방식
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),     #loss function: 모델의 정확도 측정 (최소화 해야함)
              metrics=['accuracy'])                                                     #metric: 훈련, 테스트 단계 모니터링

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

prediction = probability_model.predict(test_images)

print(prediction[0])

print(np.argmax(prediction[0]))

print(test_labels[0])
