import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

#import wandb
#from evaluate import evaluate

SMOOTH = 1e-6
#-------------------------------------------------------------------
def IoU_score(preds: Tensor, labels: Tensor, nb_classes = 8, normalize = True) :
    preds_np = np.asarray(preds.tolist())
    labels_np = np.asarray(labels.tolist())
    
    flat_pred = np.ravel(preds_np).astype('int')
    flat_label = np.ravel(labels_np).astype('int')
    
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    
    for p, l in zip(flat_pred, flat_label):
        conf_m[l, p] += 1
        
    if normalize:
        conf_m = (conf_m.astype('float') + SMOOTH) / (conf_m.sum(axis=1)[:, np.newaxis] + SMOOTH)

    if not np.isfinite(conf_m.any()) or np.isnan(conf_m.any()):
        print("INFINITE CONFUSION")
    
    # TP / (TP + FN + FP)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    
    IOU = (I + SMOOTH) / (U + SMOOTH)
    meanIOU = np.mean(IOU)

    return meanIOU, conf_m

#-------------------------------------------------------------------
def multi_acc(pred: Tensor, label: Tensor):
    _, tags = torch.max(pred, dim = 1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    return acc

#-------------------------------------------------------------------
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

#-------------------------------------------------------------------
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

#-------------------------------------------------------------------
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

#-------------------------------------------------------------------
def train_model(
        model,
        train_loader,
        val_loader,
        device,
        optimizer,
        epochs: int = 5,
        batch_size: int = 1,
        save_checkpoint: bool = True,
        amp: bool = False,
        gradient_clipping: float = 1.0,
        patience: int = 4,
        use_scheduler = True,
        debug = False):
    #. Prepare train History
    history = {"train_loss" : [], "train_IoU" : [], "train_dice" : [], 
                "val_loss" : [], "val_IoU" : [], "val_dice" : []}
    history["best_epoch"]=0
    
    early_stopper = EarlyStopper(patience=patience, min_delta=10)
    
    max_validation_score = float('-inf') #checkpoint
    
    #. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    if use_scheduler :
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)
    
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    n_train = len(train_loader)
    n_val = len(val_loader)

    #. Trainning
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0
        epoch_train_dice = 0
        epoch_accuracy = 0
        epoch_IoU = 0
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                nb_image_in_batch = images.shape[0]
                if debug :
                    print("Nb images in batch : ", nb_image_in_batch)
                    print("Nb batch : ", len(train_loader))
                    print("Nb batch : ", n_train)

                masks_pred = model(images)
                if debug :
                    print(masks_pred)
                    print(masks_pred.shape)
                mask_proba = F.softmax(masks_pred, dim=1).float()
                
                #. Loss
                loss = criterion(masks_pred, true_masks.float())
                #loss += dice_loss(mask_proba, true_masks.float(), multiclass=True)
                
                ## Accuracy
                #epoch_accuracy += multi_acc(mask_proba, true_masks)
                
                # IoU
                IoU, cm = IoU_score(mask_proba.argmax(dim=1), 
                                true_masks.argmax(dim=1))
                epoch_IoU += IoU

                # Dice
                mask_true = F.one_hot(true_masks.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_proba.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                
                epoch_train_dice += multiclass_dice_coeff(mask_pred[:, :], mask_true[:, :], reduce_batch_first=False)
                
                optimizer.zero_grad()
                #grad_scaler.scale(loss).backward() # Scale the gradients
                loss.backward() # Scale the gradients
                #grad_scaler.unscale_(optimizer)
                if gradient_clipping != 0 :
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step() # Update the model parameters
                #grad_scaler.update() # Update the scaler

                pbar.update(images.shape[0])
                #pbar.update(1)
                global_step += 1
                epoch_train_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'IoU (batch)': IoU})

        #. Log Hitory
        epoch_train_loss = (epoch_train_loss / n_train)
        epoch_train_dice = (epoch_train_dice.item() / n_train)
        #epoch_accuracy = (epoch_accuracy.item() / n_train)
        epoch_IoU = (epoch_IoU.item() / n_train)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_dice'].append(epoch_train_dice)
        #history['train_accuracy'].append(epoch_accuracy)
        history['train_IoU'].append(epoch_IoU)
        
        print(f"Epoch {epoch} --> train loss :{epoch_train_loss} | train IoU : {epoch_IoU} | train dice : {epoch_train_dice}")
        
        #print("DEBUG : epoch_train_loss", epoch_train_loss)
        
        #. Eval on train and val
        val_score, cm = evaluate(model, val_loader, device, amp)
        
        validation_score = val_score['dice']
        
        if use_scheduler :
            scheduler.step(validation_score)

        history['val_loss'].append(val_score['loss'])
        history['val_dice'].append(val_score['dice'])
        #history['val_accuracy'].append(val_score['accuracy'])
        history['val_IoU'].append(val_score['IoU'])
        
        #. Checkpoint & early stop
        if max_validation_score < validation_score :
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dice': validation_score,
                        }, f"Data/Trainned/{model.name}.pt")
            history["best_epoch"] = epoch + 1
            print(f"{validation_score} Val Dice score is better than previous {max_validation_score}, saving checkpoint epoch: ", epoch)
            max_validation_score = validation_score

        if early_stopper.early_stop(validation_score) :
            print("Early stoping at epoch ", epoch)
            break
    return history

#-------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, dataloader, device, amp: bool = False, nb_classes = 8):
    model.eval()
    num_val_batches = len(dataloader)
    total_loss = 0
    dice_score = 0
    #accuracy = 0
    total_IoU = 0
    criterion = nn.CrossEntropyLoss()
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=len(dataloader), desc='Validation round', unit='img', leave=False):
            image, true_masks = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = model(image)
            mask_proba = F.softmax(mask_pred, dim=1).float()
            
            loss = criterion(mask_pred, true_masks.float())
            #loss += dice_loss(mask_proba, true_masks.float(), multiclass=True)
            
            total_loss += loss.item()

            # Accuracy
            #accuracy += multi_acc(mask_proba, true_masks)
            
            IoU, cm = IoU_score(mask_proba.argmax(dim=1), true_masks.argmax(dim=1))
            total_IoU += IoU
            conf_m = [a+b for a, b in zip(conf_m, cm)]

            assert true_masks.min() >= 0 and true_masks.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
            # convert to one-hot format
            true_masks = F.one_hot(true_masks.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_proba.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, :], true_masks[:, :], reduce_batch_first=False)

    model.train()
    
    conf_m = np.divide(conf_m, max(num_val_batches, 1))

    dice = dice_score.item() / max(num_val_batches, 1)
    final_loss = total_loss / max(num_val_batches, 1)
    #accuracy = accuracy.item() / max(num_val_batches, 1)
    total_IoU = (IoU.item() / max(num_val_batches, 1))

    return {"loss" : final_loss, "dice" : dice, "IoU" : IoU}, conf_m

#-------------------------------------------------------------------
def Load_Checkpoint(model, device) :
    checkpoint = torch.load(f"Data/Trainned/{model.name}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

#-------------------------------------------------------------------
def infer_one_image(model, device, image) :
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        img = image.to(device=device, dtype=torch.float32)
        pred = model(img).cpu()
        mask = pred.argmax(dim=1)
    return mask

#-------------------------------------------------------------------
class EarlyStopper:
    #-------------------------------------------------------------------
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_score = float('-inf')

    #-------------------------------------------------------------------
    def early_stop(self, validation_score):
        if validation_score > self.min_validation_score:
            self.min_validation_score = validation_score
            self.counter = 0
        elif validation_score < (self.min_validation_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False