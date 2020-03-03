import pandas as pd
from PIL import Image  
import PIL
import numpy as np
from pathlib import Path

path = '../input/'

tr = pd.read_csv(path + 'train.csv', nrows = 100)
print(len(tr))
tr.head()

df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
df_train = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4' or x.split('_')[1] == '3' or 
                                                        x.split('_')[1] == '2' or x.split('_')[1] == '1')].reset_index(drop=True)
print(len(df_train))
df_train.head()

class4df_train = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')].reset_index(drop=True)
print(len(class4df_train))


def rleToMask(rleString,height,width):
  rows,cols = height,width
  rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
  rlePairs = np.array(rleNumbers).reshape(-1,2)
  img = np.zeros(rows*cols,dtype=np.uint8)
  for index,length in rlePairs:
    index -= 1
    img[index:index+length] = 255
  img = img.reshape(cols,rows)
  img = img.T
  return img
def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )

df_train['EncodedPixels'][0]
def savewithfn(index):
    fn =df_train['ImageId_ClassId'].iloc[index].split('_')[0]
    rle = df_train['EncodedPixels'][index]
    img = rleToMask(rle, 256, 1600)
    im = Image.fromarray(img)
    im.save("masks/" + fn )

for i in range(len(df_train)):
    savewithfn(i)

path = '../input/train_images/'

p = Path(path)
[x for x in p.iterdir() if x.is_dir()]

