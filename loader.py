from torch.utils.data import Dataset
from torch import Tensor
from typing import List, Tuple
from PIL import Image 
import os

class UNETLoader(Dataset):
  def __init__(self, root :str='.', img_tfs=None, mask_tfs=None):
    self.img_tfs = img_tfs  
    self.mask_tfs = mask_tfs
    self.imgs_path :str= os.path.join(root, 'images')
    self.label_path :str= os.path.join(root, 'masks')

    self.img_names :List[str]= os.listdir(self.imgs_path)
    self.label_names :List[str]= os.listdir(self.label_path)
    self.n :int= len(self.img_names)

  def __len__(self) -> int:
    return self.n
  
  def __getitem__(self, idx :int) -> Tuple[Tensor, int]:
    filename = self.img_names[idx]
    pattern = filename.split('_')[-1]
    matches = [label for label in self.label_names if label.endswith(pattern)]
    
    if len(matches) > 1:
      raise RuntimeError("More than one mask for the image")
    
    img :Image.Image= Image.open(os.path.join(self.imgs_path, self.img_names[idx]))
    mask :Image.Image= Image.open(os.path.join(self.label_path, matches[0]))

    if img.mode != 'RGB':
      img = img.convert('RGB')
    if self.img_tfs:
      img = self.img_tfs(img)
      
    if self.mask_tfs:
      mask = self.mask_tfs(mask)
    return img, mask