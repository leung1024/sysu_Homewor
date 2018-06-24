#coding=utf-8
import DataCollector
import NeuralNetwork
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 208
IMG_H = 208

train_dir = "data/train"
image_list, label_list = DataCollector.read_files(train_dir)
image_batch, label_batch = DataCollector.get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print("label: %d" % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
    coord.join(threads)


N_CLASSES = 2
IMG_H = 208
IMG_W = 208
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 15000
learning_rate = 0.0001


def run_training():
    train_dir = "data/train"
    logs_train_dir = "logs/"

    train, train_label = DataCollector.read_files(train_dir)
    train_batch, train_label_batch = DataCollector.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = NeuralNetwork.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = NeuralNetwork.losses(train_logits, train_label_batch)
    train_op = NeuralNetwork.trainning(train_loss, learning_rate)
    train_acc = NeuralNetwork.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 100 == 0:
                print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached.")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# 评估模型
from PIL import Image
import matplotlib.pyplot as plt


def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    train_dir = "C:\\Users\\panch\\Documents\\PycharmProjects\\Cats_vs_Dogs\\data\\train\\"
    train, train_label = DataCollector.read_files(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = NeuralNetwork.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        logs_train_dir = "C:\\Users\\panch\\Documents\\PycharmProjects\\Cats_vs_Dogs\\logs\\"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is %s" % global_step)
            else:
                print("No checkpoint file found")

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print("This is a cat with possibility %.6f" % prediction[:, 0])
            else:
                print("This is a dog with possibility %.6f" % prediction[:, 1])


run_training()

