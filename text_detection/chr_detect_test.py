import tensorflow as tf
import numpy as np
import cv2
import os


def train():
    target_size = (32, 32)
    batch_size = 32

    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1 / 255.))

    train_generator = generator.flow_from_directory('./samples/all/train',
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

    validation_generator = generator.flow_from_directory('./samples/all/test',
                                                         target_size=target_size,
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         shuffle=True)

    classes = len(train_generator.class_indices)

    input_tensor = tf.keras.layers.Input(shape=(32, 32, 3), name='image', dtype=tf.float32)
    base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    model = tf.keras.models.Sequential()
    model.add(base)
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dense(classes, activation='softmax', name='predictions'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    cp_path = './checkpoints/pcb_{epoch:02d}_{val_loss:.4f}_{val_acc:.2f}.h5'
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, verbose=2, save_best_only=True)
    # cb_earlystopping = tf.keras.callbacks.EarlyStopping(verbose=2, patience=50)

    history = model.fit(x=train_generator,
                        validation_data=validation_generator,
                        epochs=300,
                        callbacks=[cb_checkpoint],
                        verbose=2,
                        shuffle=True)

    return history


def test():
    model = tf.keras.models.load_model('models/pcb_all_0.0182_0.99.h5')
    base_path = './samples/test'
    items = os.listdir(base_path)
    result = open('./samples/result.txt', 'w')
    for item in items:
        input_image = cv2.resize(cv2.imread(os.path.join(base_path, item)), (32, 32), cv2.INTER_LINEAR)
        pred = model.predict(np.expand_dims(input_image, 0))
        print("{}: {}".format(item, pred[0]))
        result.write("{}: {}\n".format(item, pred[0]))
    result.close()


if __name__ == "__main__":
    pass
