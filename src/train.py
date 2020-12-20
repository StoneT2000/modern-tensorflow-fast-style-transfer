from loss import grad, loss
from model import get_vgg_model
import transform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import pandas as pd
import argparse


def parse_image(img, size=(256, 256)):
    return tf.image.resize(img, size)
def get_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    return img
def get_image_batch(paths):
    image_batch = np.zeros((len(paths), 256, 256, 3))
    for i, path in enumerate(paths):
        img = get_img(path)
        img = parse_image(img)
        image_batch[i] = img
    return image_batch

history = {
    "tl": [],
    "fr": [],
    "style": [],
    "tv": [],
}

def train(transform_net, optimizer, extractor, manager, style_target, content_paths, batch_size=32, epochs=1000):  
    """
    train the transform network using the given optimizer, vgg19 feature-extractor, and style image and paths to content images
    """
    style_targets = np.expand_dims(style_target, axis=0)
    train_size = len(content_paths)
    for epoch in range(epochs):
        # shuffle train data
        content_paths = shuffle(content_paths)
        for step in range(0, train_size, batch_size):
            batch_paths = content_paths[step:step + batch_size]
            curr_batch_size = len(batch_paths)
            
            image_batch = get_image_batch(batch_paths)
            losses, grads = grad(
              transform_net=transform_net, 
              model_feature_extractor=extractor, 
              content_targets=image_batch, 
              style_target=style_targets,
              batch_size=curr_batch_size
            )
            # use optimizer to backprop though transform net
            optimizer.apply_gradients(zip(grads, transform_net.trainable_variables))

            # get loss data and print it
            tl = losses[0] + losses[1] + losses[2]
            history['tl'].append(tl)
            history['fr'].append(losses[0])
            history['style'].append(losses[1])
            history['tv'].append(losses[2])
            optimizer_steps = optimizer.iterations.numpy()
            print("Step: {}, Total Loss: {} Feature Loss: {} Style Loss: {} TV Loss: {}".format(optimizer_steps,tl,
                                                    losses[0].numpy(), losses[1].numpy(), losses[2].numpy()))
            
            if optimizer_steps % 100 == 1:
                show_result("chicago.jpg", transform_net, True, "eval/{}.png".format(optimizer_steps), "Step {}".format(optimizer_steps))
                if manager:
                    manager.save()

        print("Epoch: {} finished".format(epoch))

def show_result(path, transform_net, extractor, style_target, save=False, savename="pic.png", msg=""):
    content_targets = get_image_batch([path])
    res = transform_net(content_targets / 255)
    feature_reconstruction_loss, style_loss, tv_loss = loss(transform_net, extractor, content_targets, np.expand_dims(style_target, 0), 1)
    loss_value = feature_reconstruction_loss + style_loss + tv_loss 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))
    plt.rcParams.update({'font.size': 14})
    plt.suptitle("Loss: {}, FR: {}, Style: {}, TV: {}\n {}".format(loss_value, feature_reconstruction_loss, style_loss, tv_loss, msg))
    plt.subplot(1,3,1)
    plt.imshow(parse_image(get_img(path)) / 255)
    plt.subplot(1,3,2)
    plt.imshow(np.clip(res[0], 0, 255) / 255)
    plt.subplot(1,3,3)
    plt.imshow(style_target / 255)
    if save:
      plt.savefig(savename)
      plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Train a Fast Style Network')
    parser.add_argument('--ckpt', type=str,
                        help='checkpoint model to use to load and start training from')
    parser.add_argument('--style', type=str,
                        help='path to style image to use for styling')
    args = parser.parse_args()

    # get the vgg19 feature extractor
    vgg_model = get_vgg_model()
    vgg_model_outputs_dir = dict([(layer.name, layer.output) for layer in vgg_model.layers])
    extractor = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model_outputs_dir)
    print("Created VGG19 Feature Extractor")

    # get the image paths
    image_paths_df = pd.read_csv("train.csv", names=['path'])
    image_paths_df['path'] = image_paths_df['path'].apply(lambda x : ('train2014/' + x))
    image_paths = image_paths_df['path'].to_numpy()

    transform_net = None
    optimizer = None

    checkpoint_path = args.ckpt

    if checkpoint_path:
        restored_model = transform.transform_net()
        restored_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        checkpoint = tf.train.Checkpoint(opt=restored_optimizer, net=restored_model)
        checkpoint.restore(checkpoint_path)
        optimizer = restored_optimizer
        transform_net = restored_model
    else: 
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        transform_net = transform.transform_net()

    # initialize checkpoint manager
    ckpt = tf.train.Checkpoint(opt=optimizer, net=transform_net)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    style = get_img(args.style)
    train(
        transform_net=transform_net, 
        extractor=extractor,
        manager=manager,
        batch_size=32, 
        epochs=int(10),
        optimizer=optimizer, 
        style_target=style,
        content_paths=image_paths,
    )

if __name__ == '__main__':
    main()