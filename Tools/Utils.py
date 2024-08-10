import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

import itertools

#-------------------------------------------------------------------
def init() :
    global categories
    categories = {  'void': [0, 1, 2, 3, 4, 5, 6],
                    'flat': [7, 8, 9, 10],
                    'construction': [11, 12, 13, 14, 15, 16],
                    'object': [17, 18, 19, 20],
                    'nature': [21, 22],
                    'sky': [23],
                    'human': [24, 25],
                    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

    global classes
    classes = ['void', 'flat', 'construction', 'object', 
                'nature', 'sky', 'human', 'vehicle']

#-------------------------------------------------------------------
def display_image_n_mask(dataset, id_list, img_size) :
    for id in id_list :
        img = dataset[id]['image']
        mask = dataset[id]['mask']
        
        img = np.transpose(img, (1, 2, 0))
        
        mask = np.transpose(mask, (1, 2, 0))
        mask = np.argmax(mask, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.resize(mask, img_size)

        fig = plt.figure(figsize=(10, 10))

        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.imshow(img)
        
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.imshow(mask)

        plt.show()

#-------------------------------------------------------------------
def superpose_image_mask_pred(image, true_mask, pred, img_size) :
    img = np.transpose(image, (1, 2, 0))
    
    mask = np.transpose(true_mask, (1, 2, 0))
    mask = np.argmax(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.resize(mask, img_size)
    pred = np.resize(pred, img_size)
    
    fig = plt.figure(figsize=(20, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    display_list = [img, mask, pred]
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_title(title[i])
        ax.imshow(display_list[i])
        ax.axis('off')
    fig.show()
    
    plt.figure(figsize = (10,10))
    plt.imshow(img)
    plt.imshow(pred, alpha=0.5)

#-------------------------------------------------------------------
def plot_model(history, title):
    loss = history['train_loss']
    val_loss = history['val_loss']
    dice = history['train_dice']
    val_dice = history['val_dice']
    IoU = history['train_IoU']
    val_IoU = history['val_IoU']

    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    fig.suptitle(title)
    
    ax[0].set(title='Loss')
    ax[0].plot(epochs, loss, label='Training')
    ax[0].plot(epochs, val_loss, label='Validation')
    ax[0].legend(loc="upper right")
    
    ax[1].set(title='dice_coeff')
    ax[1].plot(epochs, dice, label='Training')
    ax[1].plot(epochs, val_dice, label='Validation')
    ax[1].legend(loc="lower right")
    
    ax[2].set(title='Mean IoU')
    ax[2].plot(epochs, IoU, label='Training')
    ax[2].plot(epochs, val_IoU, label='Validation')
    ax[2].legend(loc="lower right")
    
    plt.close(fig)
    return fig

#----------------------------------------------------------------
def confusion_matrix(cm):
    plt.figure(figsize=(15,15))
    plot_confusion_matrix(cm, normalize=True)
    #plt.title(f'{} confusion matrix\nMean IOU: '+ str(np.round(meanIOU, 4)))
    
#----------------------------------------------------------------
def plot_confusion_matrix(cm, mean_ioU, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"{title}\nMean IoU : {np.round(mean_ioU, 4)}",fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90,fontsize=9)
    plt.yticks(tick_marks, classes,fontsize=9)
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    
    plt.tight_layout()
    plt.ylabel('True label',fontsize=9)
    plt.xlabel('Predicted label',fontsize=9)