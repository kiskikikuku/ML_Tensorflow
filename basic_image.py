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

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label],
                                         color=color))

def plot_value_array(i, prediction_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

"""
i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, prediction[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction[i], test_labels)
plt.show()                                              #0번째 원소의 이미지, 예측, 신뢰도 점수 배열

i=12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, prediction[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction[i], test_labels)
plt.show()                                              #12번째 원소의 이미지, 예측, 신뢰도 점수 배열

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, prediction[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, prediction[i], test_labels)
plt.tight_layout()
plt.show()                                               #5x3개 원소의 이미지, 예측, 신뢰도 점수 배열
"""
img = test_images[1]
print(img.shape)

img = (np.expand_dims(img,0))
print(img.shape)

prediction_single = probability_model.predict(img)
print(prediction_single)

plot_value_array(1, prediction_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(prediction_single[0])