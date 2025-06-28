import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import os
from alexnet import build_alexet 
from LeNet import build_lenet
from VGG import build_vgg
from resnet import build_resnet



(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
#Normalizing the dataset=> why divide by 255 because colour has value from 0 to 255 therefore we want in range 0to 1
x_train,x_test = x_train/255.0,x_test/255.0 

if not os.path.exists("save_models"):
    os.makedirs('save_models')

if not os.path.exists("results"):
    os.makedirs('results')


def plot_metrics(history, model_name):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_metrics.png')  # Save the figure
    plt.close()  # Avoid displaying inline during batch runs


def train_and_evaluate(model,model_name,epochs=10):
    model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        filepath = f"save_models/{model_name}.h5",monitor='val_accuracy',save_best_only = True,verbose=1
    )
    if model_name == 'LeNet':
        x_train_gray = tf.image.rgb_to_grayscale(x_train)
        x_test_gray = tf.image.rgb_to_grayscale(x_test)
        history = model.fit(x_train_gray, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1,callbacks = [checkpointing])
        test_loss, test_acc = model.evaluate(x_test_gray, y_test,verbose =1)
    else:
        history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1,callbacks = [checkpointing])
        test_loss, test_acc = model.evaluate(x_test,y_test,verbose=1)
    plot_metrics(history, model_name)
    return test_acc


# Main execution
if __name__ == "__main__":
    acc1 = train_and_evaluate(build_alexet(), 'AlexNet', epochs=10)
    acc2 = train_and_evaluate(build_lenet(), 'LeNet', epochs=10)
    acc3 = train_and_evaluate(build_vgg(), 'VGG', epochs=10)
    acc4 = train_and_evaluate(build_resnet(), 'ResNet', epochs=10)

    models = ['AlexNet', 'LeNet', 'VGG', 'ResNet']
    accuracies = [acc1, acc2, acc3, acc4]

    # Final bar chart for comparison
    plt.bar(models, accuracies, color='skyblue')
    plt.title('Final Test Accuracy by Model')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    plt.savefig('results/accuracy_comparison.png')
    plt.show()
