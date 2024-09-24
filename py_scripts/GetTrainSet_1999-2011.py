import os
import numpy as np

# Parent folder containing the subfolders with .npy files
parent_folder = 'arrays/'

#Training years 1999-2011
train_years=[i for i in range(1999,2012)]

test_years=[i for i in range(2012,2023)]

# Use a nested list comprehension to generate a list of file paths
file_paths = [
    os.path.join(root, file)
    for root, _, files in os.walk(parent_folder)
    for file in files
    if (file.endswith('.npy')) 
    and all('F'+str(exclude)[-2:] not in os.path.join(root, file) for exclude in test_years)
]

from pytictoc import TicToc

t = TicToc() ## TicToc("name")

print('Loading all relevant arrays')
t.tic()
# Load the .npy files using np.load
loaded_arrays = [np.load(file_path, allow_pickle=True) for file_path in file_paths]

t.toc()

big_array = np.concatenate(loaded_arrays, axis=0)

del loaded_arrays

t = TicToc() ## TicToc("name")

print('Sorting by timestamp')

t.tic()

#The MONTHLY REPORTING PERIOD (observation date) is now column 29
#Now Sorted by Observation Date #Do NOT Keep both Array copies
sorted_indices = np.argsort(big_array[:,29])
big_array=big_array[sorted_indices]

print('big_array shape')
print(big_array.shape)

t.toc()

t = TicToc() ## TicToc("name")

t.tic()

#Keeping only observations up to including 2011
trainset=big_array[np.where(big_array[:,29]<201201)[0]]

np.save('240214_TrainSet_1999-2011.npy', trainset)

#Observations starting from 2012 of loans that originated before 2012
restset=big_array[np.where(big_array[:,29]>=201201)[0]] 

np.save('240214_RestSet_1999-2011.npy', restset)

t.toc()

print('train set shape, rest set shape')
print(trainset.shape,restset.shape)

