import numpy as np
from skimage import io, transform
import torch, torchvision

from . import Utils

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class CSDataset(torch.utils.data.Dataset): # inheritin from Dataset class
    #-------------------------------------------------------------------
    def __init__(self, data, img_size, augmentation=None, collab_prefix=None, **kwargs) :
        super().__init__(**kwargs)
        self.img_height = img_size[0]
        self.img_width = img_size[1]
        self.data = data
        self.list_IDs = data['ID']
        self.size = len(data['ID'])
        self.collab_prefix = collab_prefix
        self.augmentation = augmentation


    #-------------------------------------------------------------------
    def __len__(self):
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return len(self.list_IDs)
        #return len(self.data) # return length (numer of rows) of the dataframe

    #-------------------------------------------------------------------
    def __getitem__(self, idx) :
        #indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #X, y = self.__data_generation(list_IDs_temp)
        X, y = self.__data_generation(idx)

        return {
            'image': torch.as_tensor(X.copy()).float().contiguous(),
            'mask': torch.as_tensor(y.copy()).long().contiguous()
        }
        
    #-------------------------------------------------------------------
    def __data_generation(self, list_IDs_temp):
        #idx = np.random.randint(0, 50000, batch_size) # ??
        
        ID = list_IDs_temp
        
        
        image_path = self.data['image_path'][ID]
        mask_path = self.data['labelID_path'][ID]

        if self.collab_prefix :
            image_path = image_path.replace("./", self.collab_prefix)
            mask_path = mask_path.replace("./", self.collab_prefix)
        
        img = torchvision.io.read_image(image_path) #ImageReadMode.RGB #ImageReadMode.GRAY
        img_mask = torchvision.io.read_image(mask_path, torchvision.io.ImageReadMode.GRAY) #ImageReadMode.RGB #ImageReadMode.GRAY

        T = torchvision.transforms.Resize((self.img_height, self.img_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        img = T(img)
        img_mask = T(img_mask)

        #img_array = img.permute(1, 2, 0).numpy()/255. # to float

        img_array = img.numpy() / 255.0
        
        mask_array = img_mask.numpy()
        img_array = np.squeeze(img_array)
        mask_array = np.squeeze(mask_array)
            
        #if self.augmentation:
        #    sample = self.augmentation(image=img, mask=img_mask)
        #    img, img_mask = sample['image'], sample['mask']

        #OneHot encode
        mask = np.zeros((8, mask_array.shape[0], mask_array.shape[1]))
        for i in range(-1, 34):
            if i in Utils.categories['void']:
                mask[0,:,:] = np.logical_or(mask[0,:,:], (mask_array==i))
            elif i in Utils.categories['flat']:
                mask[1,:,:] = np.logical_or(mask[1,:,:], (mask_array==i))
            elif i in Utils.categories['construction']:
                mask[2,:,:] = np.logical_or(mask[2,:,:], (mask_array==i))
            elif i in Utils.categories['object']:
                mask[3,:,:] = np.logical_or(mask[3,:,:], (mask_array==i))
            elif i in Utils.categories['nature']:
                mask[4,:,:] = np.logical_or(mask[4,:,:], (mask_array==i))
            elif i in Utils.categories['sky']:
                mask[5,:,:] = np.logical_or(mask[5,:,:], (mask_array==i))
            elif i in Utils.categories['human']:
                mask[6,:,:] = np.logical_or(mask[6,:,:], (mask_array==i))
            elif i in Utils.categories['vehicle']:
                mask[7,:,:] = np.logical_or(mask[7,:,:], (mask_array==i))
        
        #mask = np.resize(mask, (self.img_height * self.img_width, 8))
        mask = np.resize(mask, (8, self.img_height, self.img_width))

        if not np.isfinite(mask.any()) or np.isnan(mask.any()):
            print("INFINITE !!!!!!!!!!")

        #mask = np.argmax(mask, axis=0)
        #mask = np.expand_dims(mask, axis=0)
        #mask = np.resize(mask, (self.img_height, self.img_width))

        # To tensor ?
        return img_array, mask