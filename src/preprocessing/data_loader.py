import os
import random
import shutil
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

RAW_PATH = "data/raw/plantvillage dataset/color"
PROCESSED_PATH = "data/processed"

IMG_SIZE = (224,224)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

def create_folders():
    for split in ["train","val","test"]:
        os.makedirs(os.path.join(PROCESSED_PATH,split),exist_ok=True)

def split_data(files):
    random.shuffle(files)
    n=len(files)
    train=files[:int(n*TRAIN_SPLIT)]
    val=files[int(n*TRAIN_SPLIT):int(n*(TRAIN_SPLIT+VAL_SPLIT))]
    test=files[int(n*(TRAIN_SPLIT+VAL_SPLIT)):]
    return train,val,test

def preprocess_image(src,dst,augment=False):
    try:
        img=Image.open(src).convert("RGB")
        img=img.resize(IMG_SIZE)

        if augment:
            img=tf.keras.preprocessing.image.img_to_array(img)
            img=data_augmentation(tf.expand_dims(img,0))
            img=tf.squeeze(img).numpy()
            img=Image.fromarray(img.astype("uint8"))

        img.save(dst)

    except:
        print("Skipping corrupt image:",src)

def process_dataset():

    create_folders()

    classes=os.listdir(RAW_PATH)

    for cls in classes:

        class_path=os.path.join(RAW_PATH,cls)
        images=os.listdir(class_path)

        train,val,test=split_data(images)

        for split,files in zip(["train","val","test"],[train,val,test]):

            save_dir=os.path.join(PROCESSED_PATH,split,cls)
            os.makedirs(save_dir,exist_ok=True)

            for img in tqdm(files):

                src=os.path.join(class_path,img)
                dst=os.path.join(save_dir,img)

                if split=="train":
                    preprocess_image(src,dst,augment=True)
                else:
                    preprocess_image(src,dst,augment=False)

if __name__=="__main__":
    process_dataset()
