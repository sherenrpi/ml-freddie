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
    and all('F'+str(exclude)[-2:] not in os.path.join(root, file) for exclude in train_years)
]

from pytictoc import TicToc

t = TicToc() ## TicToc("name")

print('Loading all relevant arrays')
t.tic()
# Load the .npy files using np.load
loaded_arrays = [np.load(file_path, allow_pickle=True) for file_path in file_paths]

t.toc()

first_big_array = np.concatenate(loaded_arrays, axis=0)

print('first_big_array shape')
print(first_big_array.shape)

del loaded_arrays

# Add the observations starting in 2012 of loans that originated before 2012

rest_array=np.load('240214_RestSet_1999-2011.npy', allow_pickle=True)

print('rest_array shape')
print(rest_array.shape)

big_array = np.concatenate((first_big_array,rest_array), axis=0)

del first_big_array,rest_array

t = TicToc() ## TicToc("name")

print('Sorting by timestamp')

t.tic()

#Now Sorted by Observation Date #Do NOT Keep both Array copies
sorted_indices = np.argsort(big_array[:,29])
big_array=big_array[sorted_indices]

print('big_array shape')
print(big_array.shape)

t.toc()

t = TicToc() ## TicToc("name")

t.tic()

#There should be NO observations before 2012
trainset=big_array[np.where(big_array[:,29]<201201)[0]]

print('There should be NO observations before 2012')
print(trainset.shape)

# #Observations starting from 2012 of all loans
# fullset=big_array[np.where(big_array[:,29]>=201201)[0]] 

# np.save('240109_FullSet_2012-2022.npy', fullset)

t.toc()

#print('Full set after 2012 shape')
#print(fullset.shape)

#del fullset

#Tune Set 2012-2013
tuneset=big_array[np.where((big_array[:,29]>=201201) & (big_array[:,29]<201401))[0]]

np.save('240215_TuneSet_2012-2013.npy', tuneset)


# #240104 I want to separate the observations by year
# for y in [i for i in range(2014,2023)]:
    
#     testyear=int(str(y)+'01')
#     nextyear=int(str(y+1)+'01')

#     testset_y=big_array[np.where((big_array[:,29]>=testyear) & (big_array[:,29]<nextyear))[0]]

#     np.save(f'240215_set_year_{y}.npy', testset_y)
    
#     print('year',y)
#     print('test set shape')
#     print(testset_y.shape)

