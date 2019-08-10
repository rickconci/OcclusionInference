import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np 

csv_path = "/home/riccardo/Desktop/150k_bw_ro/digts/digts.csv"
train_image_paths = "/home/riccardo/Desktop/150k_bw_ro/digts/train/orig/"    
test_image_paths = "/home/riccardo/Desktop/150k_bw_ro/digts/test/orig/"
train_data_size = len(os.listdir(train_image_paths))
test_data_size = len(os.listdir(test_image_paths))

train_test_type = 'train'

data = pd.read_csv(csv_path, header=None)
train_data = data.iloc[:train_data_size, :]
test_data = data.iloc[train_data_size:, :]

if train_test_type=='train':
    digt_list_train = train_data.iloc[:, [5,21]].values.astype(int)
    cols_train = train_data.iloc[:, [12, 32]].values/255
    cols_train = cols_train.astype(int)
    x_back = train_data.iloc[:, 6].values.astype(float)
    y_back = train_data.iloc[:, 7].values.astype(float)
    x_front = train_data.iloc[:, 22].values.astype(float)
    y_front = train_data.iloc[:, 23].values.astype(float)
elif train_test_type=='test':
    digt_list_test = test_data.iloc[:, [5,21]].values.astype(int)
    cols_test = test_data.iloc[:, [12, 32]].values/255
    cols_test = cols_test.astype(int)
    x_back = test_data.iloc[:, 6].values.astype(float)
    y_back= test_data.iloc[:, 7].values.astype(float)
    x_front = test_data.iloc[:, 22].values.astype(float)
    y_front = test_data.iloc[:, 23].values.astype(float)
       
distance = np.sqrt((x_front - x_back)**2 + (y_front - y_back)**2)

print(np.sum(distance > 0.24)/len(distance)*100)

plt.hist(distance, bins=30)
plt.ylabel('Probability');
plt.savefig('hist_distances.png')