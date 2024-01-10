from torch.utils.data import Dataset
from skimage import io
import os
import cv2

class binary_class(Dataset):
    def __init__(self,path,data,transform =None):
        self.path = path
        self.folders = data
        self.transforms = transform
    
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.path,'images/',self.folders[index])
        img = io.imread(img_path)[:,:,:3].astype('float32')
        img_id = self.folders[index]
        augmented = self.transforms(image=img)
        img = augmented['image']
        return (img, img_id)