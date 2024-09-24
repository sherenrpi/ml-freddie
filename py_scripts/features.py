import numpy as np
import pandas as pd
import pickle


#TODO Need a check that files exist and an Error message otherwise!!!

with open('sellers_1999_2023_dict.pkl', 'rb') as f:
    bank_dict = pickle.load(f)   
    
with open('servicers_1999_2023_dict.pkl', 'rb') as f:
    servicer_dict = pickle.load(f)  
    
#Class for preparing raw Origination features
class OriginationFeatures:
    def __init__(self,df):
        self.df = df

#Freatures Description from the User Guide Release 38
#The Origination Data is read in it's entirety
#Columns are preprocessed
#Three columns that provide informations on dates are later dropped
#origdf.drop(columns=[1,3,26],inplace=True)
#
#Each column will be given a name from the User guide
#To help with later backtracking of feature meaning

    def preprocess_features(self):
        
        #CREDIT SCORE
        #A number, prepared by third parties, 
        #summarizing the borrower’s creditworthiness, 
        #which may be indicative of the likelihood 
        #that the borrower will timely repay future obligations.
        self.df.rename(columns={0:'CREDIT SCORE'}, inplace=True)

        #FIRST PAYMENT DATE
        #This column will be dropped
        #self.df.rename(columns={1:'FIRST PAYMENT DATE'}, inplace=True)
        
        #FIRST TIME HOMEBUYER FLAG
        #•Y = Yes
        #•N = No
        #•9 = Not Available or Not Applicable
        ##Transform strings to numbers
        self.df[self.df.columns[2]]=self.df[self.df.columns[2]].replace({'Y': 0, 'N': 1, '9': 9})
        self.df.rename(columns={2:'FIRST TIME HOMEBUYER FLAG'}, inplace=True)
        
        #MATURITY DATE
        #This column will be dropped
        #self.df.rename(columns={3:'MATURITY DATE'}, inplace=True)
        
        #METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION
        #Space (5) = Indicates that the area in which the mortgaged property is located is 
        #a) neither an MSA nor a Metropolitan Division, or b) unknown.
        #Space will appear as NaNs in the DF
        #Since the MSA codes range from 10180.0 to 49740.0
        #Replace nan with zeros
        self.df[self.df.columns[4]]=self.df[self.df.columns[4]].fillna(0)
        self.df.rename(columns={4:'MSA CODE'}, inplace=True)
        
        #MORTGAGE INSURANCE PERCENTAGE (MI %)
        #•1% - 55%
        #•0 = No MI
        #•999 = Not Available
        self.df.rename(columns={5:'MI PERCENTAGE'}, inplace=True)
        
        #NUMBER OF UNITS
        #•1 = one-unit
        #•2 = two-unit
        #•3 = three-unit
        #•4 = four-unit
        #•99 = Not Available
        self.df.rename(columns={6:'NUMBER OF UNITS'}, inplace=True)
        
        #OCCUPANCY STATUS
        #• P = Primary Residence
        #• I = Investment Property
        #• S = Second Home
        #• 9 = Not Available
        # Transform strings to numbers
        self.df[self.df.columns[7]]=self.df[self.df.columns[7]].replace({'P': 0, 'I': 1,'S':2, '9': 9})
        self.df.rename(columns={7:'OCCUPANCY STATUS'}, inplace=True)
        
        #ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)
        #2018Q1 and prior:
        #•6% - 200%
        #•999 = Not Available
        #2018Q2 and later:
        #•1% - 998%
        #•999 = Not Available
        self.df.rename(columns={8:'ORIGINAL CLTV RATIO'}, inplace=True)
        
        #ORIGINAL DEBT-TO-INCOME (DTI) RATIO
        #•0%<DTI<=65%
        #•999 = Not Available
        self.df.rename(columns={9:'Original DTI Ratio'}, inplace=True)
        
        #ORIGINAL UPB
        self.df.rename(columns={10:'ORIGINAL UPB'}, inplace=True)
        
        #ORIGINAL LOAN-TO-VALUE (LTV)
        #2018Q1 and prior:
        #•6% - 105%
        #•999 = Not Available
        #2018Q2 and later:
        #•1% - 998%
        #•999 = Not Available
        self.df.rename(columns={11:'ORIGINAL LTV RATIO'}, inplace=True)
        
        #ORIGINAL INTEREST RATE
        self.df.rename(columns={12:'ORIGINAL INTEREST RATE'}, inplace=True)
        
        # CHANNEL
        #• R = Retail
        #• B = Broker
        #• C = Correspondent
        #• T = TPO Not Specified
        #• 9 = Not Available
        # Transform strings to numbers
        self.df[self.df.columns[13]]=self.df[self.df.columns[13]].replace({'R': 0, 'B': 1,'C':2,'T':3, '9': 9})
        self.df.rename(columns={13:'CHANNEL'}, inplace=True)
        
        #PREPAYMENT PENALTY MORTGAGE (PPM) FLAG
        #• Y = PPM
        #• N = Not PPM
        # Transform strings to numbers
        self.df[self.df.columns[14]]=self.df[self.df.columns[14]].replace({'Y': 0, 'N': 1})
        self.df.rename(columns={14:'PPM FLAG'}, inplace=True)
        
        #AMORTIZATION TYPE
        # • FRM – Fixed Rate Mortgage
        # • ARM – Adjustable Rate Mortgage
        # Transform strings to numbers
        self.df[self.df.columns[15]]=self.df[self.df.columns[15]].replace({'FRM': 0, 'ARM': 1})
        self.df.rename(columns={15:'AMORTIZATION TYPE'}, inplace=True)
        
        #PROPERTY STATE
        #States and territories (PR: Puert Rico, GU: Guam, VI: Virgin Islands)
        states_terr = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY','PR','GU','VI']
        state_dict={}
        for i,s in enumerate(states_terr):
            state_dict[s]=i
        # Transform strings to numbers
        self.df[self.df.columns[16]]=self.df[self.df.columns[16]].replace(state_dict)
        self.df.rename(columns={16:'PROPERTY STATE'}, inplace=True)
        
        # PROPERTY TYPE
        #• CO = Condo
        #• PU = PUD
        #• MH = Manufactured Housing
        #• SF = Single-Family
        #• CP = Co-op
        #• 99 = Not Available
        # Transform strings to numbers
        self.df[self.df.columns[17]]=self.df[self.df.columns[17]].replace({'CO':0,'PU':1,'MH':2,'SF':3,'CP':4,'99':99})
        self.df.rename(columns={17:'PROPERTY TYPE'}, inplace=True)
        
        #POSTAL CODE
        self.df.rename(columns={18:'POSTAL CODE'}, inplace=True)
        
        #LOAN SEQUENCE NUMBER
        self.df.rename(columns={19:'Loan_ID'}, inplace=True)
        
        #LOAN PURPOSE
        #• P = Purchase
        #• C = Refinance - Cash Out
        #• N = Refinance - No Cash Out
        #• R = Refinance - Not Specified
        #• 9 =Not Available
        # Transform strings to numbers
        self.df[self.df.columns[20]]=self.df[self.df.columns[20]].replace({'P': 0, 'C': 1,'N':2,'R':3,'9':9})
        self.df.rename(columns={20:'LOAN PURPOSE'}, inplace=True)
        
        #ORIGINAL LOAN TERM
        self.df.rename(columns={21:'ORIGINAL LOAN TERM'}, inplace=True)
        
        #NUMBER OF BORROWERS
        #2018Q1 and prior:
        # •01 = 1 borrower
        # •02 = > 1 borrowers
        # •99 = Not Available
        # 2018Q2 and later:
        # •01 = 1 borrower
        # •02 = 2 borrowers
        # •03 = 3 borrowers
        # …
        # •09 = 9 borrowers
        # •10 = 10 borrowers
        # •99 = Not Available
        self.df.rename(columns={22:'NUMBER OF BORROWERS'}, inplace=True)
        
        #SELLER NAME
        #Name of the seller, or “Other Sellers”
        #A set of all Sellers from 1999 to 2023
        #Obtained from pre-analysis 
        self.df[self.df.columns[23]]=self.df[self.df.columns[23]].replace(bank_dict)
        self.df.rename(columns={23:'SELLER NAME'}, inplace=True)
        
        #SERVICER NAME
        #Name of the servicer, or “Other Servicers”
        #A set of all Servicers from 1999 to 2023
        #Obtained from pre-analysis  
        self.df[self.df.columns[24]]=self.df[self.df.columns[24]].replace(servicer_dict)
        self.df.rename(columns={24:'SERVICER NAME'}, inplace=True)
        
        #SUPER CONFORMING FLAG –
        # For mortgages that exceed conforming loan limits
        # with origination dates on or after 10/1/2008 
        # and were delivered to Freddie Mac on or after 1/1/2009
        #• Y = Yes
        # • Space (1) = Not Super Conforming --> nan
        self.df[self.df.columns[25]]=self.df[self.df.columns[25]].fillna(0)
        self.df[self.df.columns[25]]=self.df[self.df.columns[25]].replace({'Y':1})
        self.df.rename(columns={25:'SUPER CONFORMING FLAG'}, inplace=True)
        
        #PRE-RELIEF REFINANCE LOAN SEQUENCE NUMBER
        #This column will be dropped
        #self.df.rename(columns={26:'PRE-RELIEF REFINANCE LOAN SEQUENCE NUMBER'}, inplace=True)
        
        #PROGRAM INDICATOR
        #The indicator that identifies if a 
        #loan participates in 
        #Freddie Mac programs listed in the valid values.
        #H = Home Possible
        #F = HFA Advantage
        #R= Refi Possible
        #9 = Not Available or Not Applicable
        # Transform strings to numbers
        self.df[self.df.columns[27]]=self.df[self.df.columns[27]].replace({'H': 0, 'F': 1,'R':2,'9':9})
        self.df.rename(columns={27:'PROGRAM INDICATOR'}, inplace=True)
        
        #RELIEF REFINANCE INDICATOR
        #Populated only for loans where the Relief Refinance Indicator is set to Y. 
        #All other loans will be blank.-->NaN
        self.df[self.df.columns[28]]=self.df[self.df.columns[28]].fillna(0)
        self.df[self.df.columns[28]]=self.df[self.df.columns[28]].replace({'Y':1})
        self.df.rename(columns={28:'RELIEF REFINANCE INDICATOR'}, inplace=True)
        
        #PROPERTY VALUATION METHOD
        #The indicator denoting which method was used to obtain a property appraisal, if any.
        #Populated for loans originated on or after 1/1/2017.
        #1 = ACE Loans
        #2 = Full Appraisal
        #3 = Other Appraisals (Desktop, driveby, external, AVM)
        #4 = ACE + PDR
        #9 = Not Available
        #Already Numeric
        self.df.rename(columns={29:'PROPERTY VALUATION METHOD'}, inplace=True)
        
        # INTEREST ONLY INDICATOR (I/O INDICATOR) - 
        # The indicator denoting whether the loan only requires interest payments 
        # for a specified period beginning with the first payment date.
        #Transform strings to numbers
        self.df[self.df.columns[30]]=self.df[self.df.columns[30]].replace({'Y': 0, 'N': 1})
        self.df.rename(columns={30:'INTEREST ONLY INDICATOR'}, inplace=True)
        
        # MI CANCELLATION INDICATOR:
        #The indicator denoting if the mortgage insurance has been reported 
        #as cancelled after the time of Freddie Mac's purchase of the mortgage loan.
        #Transform strings to numbers
        self.df[self.df.columns[31]]=self.df[self.df.columns[31]].replace({'Y': 0, 'N': 1,'7':7,'9':9})
        self.df.rename(columns={31:'MI CANCELLATION INDICATOR'}, inplace=True)
        
        
        #Adding New Feature: Calculatin Value at Origination
        #Based on Loan Amount (Original UPB) and reported LTV
        #ORIGINAL LOAN-TO-VALUE (LTV) is 999 When Not Available
        #Setting the Value to 999 when LTV not available, too
        
        self.df['ORIGINAL VALUE']=np.where(self.df['ORIGINAL LTV RATIO'] != 999, self.df['ORIGINAL UPB'] / self.df['ORIGINAL LTV RATIO'], 999)      
        
        return self.df  # Return the modified DataFrame
 
    

#Class for preparing raw Perfromance features
class PerformanceFeatures:
    def __init__(self,df):
        self.df = df
#Freatures Description from the User Guide Release 38
#The Performance Data is read only partially:
#columns_2_use=[0,1,2,3,4,5,7,10,11,23,24,31]

    def clds_converter(value):
        if value == 'RA': 
            return -1.0
        elif value == 'XX':
            return -2.0
        try:
            return int(value)
        except ValueError:
            return None  # This is for space (' ') or any non-integer values


    def preprocess_features(self):

        #LOAN SEQUENCE NUMBER
        #Important to have the same column name
        #As in the origination df
        #The two dfs will be merged on Loan ID
        self.df.rename(columns={0:'Loan_ID'}, inplace=True)
        
        #MONTHLY REPORTING PERIOD
        #Importnat for later dividing the time series
        #Into Train/Tune/Test sets
        self.df.rename(columns={1:'MONTHLY REPORTING PERIOD'}, inplace=True)
        
        #Index 2 (Column 3)
        #CURRENT ACTUAL UPB
        #Calculation: (interest bearing UPB)
        #+ (non-interest bearing UPB)
        self.df.rename(columns={2:'CURRENT ACTUAL UPB'}, inplace=True)
        
        #Index 3 (column 4)
        #CURRENT LOAN DELINQUENCY STATUS
        # • XX = Unknown
        # • 0 = Current, or less than 30 days past due
        # • 1 = 30-59 days delinquent
        # • 2 = 60 – 89 days delinquent
        # • 3 = 90 – 119 days delinquent
        # • And so on…
        # • RA = REO Acquisition
        # • Space (3) = Unavailable
        #The columns are labeled by the index number,so index works as a label

        def clds_converter(value):
            if value == 'RA': 
                return -1.0
            elif value == 'XX':
                return -2.0
            try:
                return int(value)
            except ValueError:
                return None  # This is for space (' ') or any non-integer values


        self.df[3]=self.df[3].apply(clds_converter)
        self.df.rename(columns={3:'CLDS'}, inplace=True)
        
        #Index 4 (column 5)
        #LOAN AGE - The number of scheduled payments from
        #the time the loan was originated or modified up 
        #to and including the performance cutoff date.
        #Calculation: ((Monthly Reporting Period) - Loan First Payment Date (MM/YY)) +1 month
        self.df.rename(columns={4:'LOAN AGE'}, inplace=True)
        
        #Index 5 (column 6)
        #REMAINING MONTHS TO LEGAL MATURITY
        self.df.rename(columns={5:'REMAINING MONTHS'}, inplace=True)
        
        #Index 7 (Column 8)
        #MODIFICATION FLAG – 
        #For mortgages with loan modifications, 
        #indicates that the loan has been modified.
        #•Y = Current Period Modification
        #•P = Prior Period Modification
        #•Null = Not Modified
        #Most loans will have Null
        self.df[7]=self.df[7].fillna(-1)
        self.df[7]=self.df[7].replace({'Y': 1, 'P': 0})
        self.df.rename(columns={7:'MODIFICATION FLAG'}, inplace=True)
        
        #Index 10 (Column 11)
        #CURRENT INTEREST RATE - Reflects the current
        #interest rate on the mortgage note, 
        #taking into account any loan modifications.
        self.df.rename(columns={10:'CURRENT INTEREST RATE'}, inplace=True)

        #Index 11 (Column 12)
        #CURRENT NON-INTEREST BEARING UPB - 
        #The non-interest-bearing portion of the UPB for a given mortgage.
        self.df.rename(columns={11:'CURRENT NON-INTEREST BEARING UPB'}, inplace=True)
        
        #Index 23 (Column 24)
        #STEP MODIFICATION FLAG
        #• Y = Current Period Step Mod
        #• N = Current Period Non-Step Mod
        #• Space (1) = Not a Current Period Mod --> Set to -1
        self.df[23]=self.df[23].fillna(-1)
        self.df[23]=self.df[23].replace({'Y': 1, 'N': 0})
        self.df.rename(columns={23:'STEP MODIFICATION FLAG'}, inplace=True)
        
        #Index 24 (Column 25)
        #PAYMENT DEFERRAL – 
        #A flag indicating a loan has been granted a 
        #Payment Deferral in the current or prior period.
        #• Y = Current Period 
        #• P = Prior Period
        #• Null = Not Payment Deferral
        self.df[24]=self.df[24].fillna(-1)
        self.df[24]=self.df[24].replace({'Y': 1, 'P': 0})
        self.df.rename(columns={24:'PAYMENT DEFERRAL'}, inplace=True)
        
       
        #Index 31 (Column 32)
        #INTEREST BEARING UPB
        #The current interest bearing UPB of the modified mortgage.
        #My Note: Only populated if MODIFICATION FLAG is Not Null
        self.df[31]=self.df[31].fillna(-1)
        self.df.rename(columns={31:'MODIFIED INTEREST BEARING UPB'}, inplace=True)
        
        return self.df  # Return the DataFrame
        
        
        
    #Function to define the Target based on CLDS    
    def get_target(self):
    
        target = []
        pastever1 = []
        pastever2 = []
        pastever3 = []

        #Let's make sure we always use the correct clds column!
        #The columns index of the clds is 3
        dfarray = self.df['CLDS'].to_numpy()

        # Helper function to check conditions
        def check_condition(clds, threshold):
            return (clds >= threshold).any()

        # Normal case for longer loan history
        if len(self.df)>24:
            for k in range(len(self.df)-24):
                    future_clds = dfarray[k + 1:k + 25]

                    if check_condition(future_clds, 4) or (future_clds <= -1).any():
                        target.append(0)
                    else:
                        target.append(1)
                        #Exclude the current reporting period clds,
                        #since it's included in the clds column itself
                    past_clds = dfarray[:k]

                    pastever1.append(int(check_condition(past_clds, 1)))
                    pastever2.append(int(check_condition(past_clds, 2)))
                    pastever3.append(int(check_condition(past_clds, 3)))

            #Drop the last 24 rows as they don't have a target associated with them
            self.df.drop(self.df.tail(24).index,
                inplace = True)

        # Special case for short loan history <=24 months
        # Only keep the first row, and look into the following
        # Rows for target
        else:
            k=0
    #         print('Short history',len(df))
            future_clds = dfarray[k+1:]
    #         print(future_clds)
            if check_condition(future_clds, 4) or (future_clds <= -1).any():
                target.append(0)
            else:
                target.append(1)
            #Exclude the current reporting period clds,
            #since it's included in the clds column itself
            past_clds = dfarray[:k]

    #         print(past_clds)
            pastever1.append(int(check_condition(past_clds, 1)))
            pastever2.append(int(check_condition(past_clds, 2)))
            pastever3.append(int(check_condition(past_clds, 3)))

            #Drop all but the first row, 
            #as the remaining rows don't have a target associated with them
            self.df.drop(self.df.tail(len(self.df)-1).index,
                inplace = True)


        # Set new columns for past lds features
        self.df['PAST CLDS 1'] = pastever1
        self.df['PAST CLDS 2'] = pastever2
        self.df['PAST CLDS 3'] = pastever3
        # Set new column containing the target
        self.df['TARGET'] = target

        return self.df
