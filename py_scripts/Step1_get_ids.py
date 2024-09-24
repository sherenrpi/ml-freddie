import numpy as np
import pandas as pd
from zipfile import ZipFile
import json

from pytictoc import TicToc
t = TicToc() ## TicToc("name")


#The purpose of this script is to preselect
#The loans and to save their IDs
#We will look at the length of the loan's 
#observation data history in teh 'svcg' file
#And the loan's final status (zero-balance code= ZBL)

#We will keep all loans that are NOT Matured (ZBL!=01)
#We will keep the loans that are matured (ZBL!=01) ONLY when
# there is at least 24 months of observation history data available

import argparse

# Initialize parser
# Step 2: Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Script determines all relevant loan IDs in a given range of years')
# Step 3: Define a command-line argumen
parser.add_argument('years', nargs='+', type=int, help='Provide a list of years in [1999,2022] as "*.py 2005 2006 2007 2008"')
# parser.add_argument('years',type=int, help='Provide years in [1999,2022] as a list like e.g. [2005,2009]')
# Step 4: Parse the command-line arguments
args = parser.parse_args()

if not args.years:
    parser.error('At least one year must be provided.')

#Step 5: Access and use the argument value
print(f'The argument you provided is: {args.years}')

years=args.years

print(years)


t.tic()

for y in years:
    print(y)

    zip_file = ZipFile(f'zip_data/sample_{y}.zip')

    text_file=[text_file for text_file in zip_file.infolist() if 'svcg' in text_file.filename]

    df = pd.read_csv(zip_file.open(text_file[0].filename), sep="|",header=None
          ,usecols =[0,8] #0: Loan_ID, 8: Zero Balance Code
          , dtype=object  #Loading everything as str, faster for mixed types     
        )

    
    #Each row is a monthly observation
    #Labeled by loan_id
    #We want only the unique loan identifiers
    loanids=df.drop_duplicates(keep='first')


    perfromance_rows_dict={}

    short_paid_off={}

    #Now, go through the ids and check history length and ZBL
    for li in loanids.to_dict(orient='list')[0]:
        #loans with histories shorter than 25 months
        #AND paid off --> will not be considered 
        #(but we'll save them for reference)
        if len(df.index[df[0]==li].tolist())<25 and df.loc[df[0] == li, 8].iloc[-1]=='01':
            short_paid_off[li]=[int(y),df.index[df[0]==li].tolist()]

        else:
            perfromance_rows_dict[li]=[int(y),df.index[df[0]==li].tolist()]

    with open(f"{y}_kept_IDs.json", "w") as outfile:
        json.dump(perfromance_rows_dict, outfile)
            
    with open(f"{y}_short_IDs.json", "w") as outfile:
        json.dump(short_paid_off, outfile)


t.toc()
