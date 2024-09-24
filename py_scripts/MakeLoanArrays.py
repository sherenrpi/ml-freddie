import argparse

# Initialize parser
# Step 2: Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Script produces a Complete Array for a given loan ID')
# Step 3: Define a command-line argumen
parser.add_argument('origyear',type=int, help='Provide origination year in [1999,2022]')
# Step 4: Parse the command-line arguments
args = parser.parse_args()


#Step 5: Access and use the argument value
print(f'The argument you provided is: {args.origyear}')

y=args.origyear

print(y)


from zipfile import ZipFile
import numpy as np
import pandas as pd

from pytictoc import TicToc

import json

from features import OriginationFeatures,PerformanceFeatures 
#PerformanceFeatures class contains the get_target(your_dataframe) function  
#That designs additional features based on the CLDS
# Example usage:
# result_df = get_target(your_dataframe)


# Opening JSON file
with open(f'all_json_files/{y}_kept_IDs.json') as json_file:
    ids_data = json.load(json_file)
    
#Keep the monthly performance date to then later 
#separate into train and test sets by the year 
columns_2_use=[0,1,2,3,4,5,7,10,11,23,24,31]

t = TicToc() ## TicToc("name")

#For every id make an array of origination and performance data history
#Read in the file ONLY for that ID !!! Important for memory

t.tic()

for i in ids_data:

    year=ids_data[i][0]
    skiprows=ids_data[i][1][0]
    nrows=len(ids_data[i][1])

    zip_file = ZipFile(f'zip_data/sample_{year}.zip')

    text_file=[text_file for text_file in zip_file.infolist() if 'svcg' in text_file.filename]

    df = pd.read_csv(zip_file.open(text_file[0].filename), sep="|",header=None
                     ,skiprows=skiprows,nrows=nrows 
                     ,usecols =columns_2_use
                          )
    #Prepare raw data
    df=PerformanceFeatures(df).preprocess_features()
    
    #df=PerformanceFeatures(df).get_target()
    
    #Already included in the new features.py
    #df.rename(columns={0:'Loan_ID'}, inplace=True)
    
    #Load & Prepare origination Data

    origtext_file=[text_file for text_file in zip_file.infolist() if 'orig' in text_file.filename]

    origdf = pd.read_csv(zip_file.open(origtext_file[0].filename), sep="|",header=None,low_memory=False)
    origdf=OriginationFeatures(origdf).preprocess_features()
    origdf.drop(columns=[1,3,26],inplace=True)
#     origdf.rename(columns={19:'Loan_ID'}, inplace=True) #Already included in the new features.py


    #2402 NEW Feature: Current LTV available only for loans after 2014
    #Estimating current LTV from the Origination LTV assuming Value remains constant
    #(Crude approximation)
    #The original VALUE column has been already created during origination df preprocessing
    #Now, merge on ID first
    #this will populate the value for every reporting month of the performance df 
    #Then calculate Current LTV based on current UPB and original Value
    #When VALUE is not available (because original LTV was not avaiolabe)
    #It is set to 999, do the same for the New Extimated Current LTV
    completedf=origdf.merge(df,on='Loan_ID',how='inner')
    
    completedf['ESTIMATED CURRENT LTV'] = np.where(completedf['ORIGINAL VALUE'] != 999, completedf['CURRENT ACTUAL UPB']/completedf['ORIGINAL VALUE'], 999)
   
    #We want the target to be in the very last column!
    completedf=PerformanceFeatures(completedf).get_target()


    #Every observation at a given monthly reporting period
    #now has a complete feature vector with associated target 
    #
    completedf.drop(columns='Loan_ID',inplace=True)
    
    np.save(f'arrays/{i}_origperform.npy', completedf.to_numpy())
   
    
# Save column names to a CSV file
completedf.columns.to_series().to_csv('column_names.csv', index=False, header=False)
    
t.toc()
