import numpy as np 
import cv2
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
latent_dim = 100
import pandas as pd
from sentenceGenerate import generation_txt
from main_vocab import build_vocab
import tensorflow as tf
import csv
# Set log level to ERROR to ignore warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Alternatively, you can use this line to suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

 
num_samples = 50
input_image_path = "../Data/images/"
path ="../Data/multimodal_train.csv"
txt_vocab = "../data/FakeNews.txt"

data = pd.read_csv(path)
data = data[data['6_way_label'] == 3]
text = data.clean_title
img_id = data.id
label = data['6_way_label']

New_img =[]
New_txt =[]
New_labels = []

if not os.path.exists("../Data/GAN_img/"):
  os.makedirs("../Data/GAN_img/")
# Genration Image 
# Load the saved generator model
loaded_generator = load_model('gan_generator_model.h5')


# Generate and save images using the loaded model
def generate_and_save_images(input_image, num_samples, output_prefix):
    r, c = num_samples, 1
    label =1
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    input_images = np.array([input_image] * num_samples)
    gen_imgs = loaded_generator.predict([noise, input_images])

    # Rescale images 
    gen_imgs = 0.5 * gen_imgs + 0.5

    for idx, img in enumerate(gen_imgs):
        plt.savefig(f"../Data/GAN_img/{output_prefix}{idx}.jpg")
        New_img.append(f"{output_prefix}{idx}.jpg")
        New_labels.append(label)
        plt.close()

def load_input_image(image_path, target_size=(28, 28), grayscale=True): 
    img = cv2.imread(image_path)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, target_size)
    img = img / 127.5 - 1.0
    img = np.expand_dims(img, axis=-1)
    return img


for i in img_id: 
    _path = input_image_path+str(i)+'.jpg'
    if os.path.exists(_path):
        print( "generation image now for ",str(i))
        input_image = load_input_image(input_image_path+str(i)+'.jpg') 
        output_prefix = "loaded_model_"+str(i)
        generate_and_save_images(input_image, num_samples, output_prefix)
        

#Genration Text
News_file = "data/FakeNews.txt"
def short_sentence(text):
    words = text.split()
    if len(words) <= 2: 
        return True
    return False

def generate_and_save_texts(txt):
    print(txt)
    if (short_sentence(txt) == True):
        txt = txt+" "
        txt = txt *2

    with open(News_file, "w") as file:
            for _ in range(len(txt)):
                file.write(txt+"\n")
    
    build_vocab()
    
    generation_txt(num_samples)
    

    txt_path ="save_generator/text_sentenceGenerate.txt"
    with open(txt_path, "r") as file:
        for line in file:
            New_txt.append(line)


print( "generation text ....."  ) 
for i in range(len(img_id)): 
    _path = input_image_path+str(img_id[i])+'.jpg'
    if os.path.exists(_path):        
            generate_and_save_texts(text[i])

# Remove the null text from the results 
# Find all indices of the value 2
value_to_find = '\n'
indices = [index for index, element in enumerate(New_txt) if element == '\n' or element == ' ' ]
print("Indices of value", value_to_find, ":", indices)

# Remove elements at the specified indices from all three lists
New_img2 = [element for index, element in enumerate(New_img) if index not in indices]
New_txt2 = [element for index, element in enumerate(New_txt) if index not in indices]
New_labels2 = [element for index, element in enumerate(New_labels) if index not in indices]


#Save the results:
def merge_lists_to_csv(filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header
        header = ["id", "clean_title", "6_way_label"]
        csv_writer.writerow(header)
        
        # Assuming all lists have the same length
        for i in range(len(New_img2)):
            csv_writer.writerow([New_img2[i], New_txt2[i], New_labels2[i]])
            

merge_lists_to_csv('output_Genration_Dataset.csv')
