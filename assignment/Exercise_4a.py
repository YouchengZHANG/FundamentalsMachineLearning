
## Exercise 4a

# 1 Comment on your and other's solution to Exercise 4

# 2 Red Cards Study
# https://www.nature.com/news/%20crowdsourced-research-many-hands-make-tight-work-1.18508
# https://osf.io/gvm2z/files/
# https://osf.io/2prib/
# https://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb
# https://notebook.community/cmawer/pycon-2017-eda-tutorial/notebooks/1-RedCard-EDA/3-Redcard-Dyads



# 2.1 Loading and Cleaning the Data (10 points)
from operator import length_hint
import pandas as pd                                    
import numpy as np                             
import os       
import matplotlib.pyplot as plt               
filename=os.path.join('CrowdstormingDataJuly1st.csv') 
df0 = pd.read_csv(filename)
df0.shape
df0.head()
df0.columns.values
df0['games']
np.unique(df0['games'])

# What do the feature names (e.g. column games) stand for?
# games: the number of games in the player-referee dyad
# player–referee dyads including the number of matches players and referees encountered each other 
# the number of red cards given to a player by a particular referee throughout all matches the two encountered each other.

# Which irrelevant features might be dropped?
# birthday, ties, defeats, goals,yellowCards, nIAT, seIAT, nExp, seExp, victories, Alpha_3, photoID(unless want to improve color rating)
irfeature = ['player','birthday','ties','defeats','goals','yellowCards', 'nIAT', 'seIAT', 'nExp', 'seExp', 'victories', 'Alpha_3', 'photoID']
df = df0.drop(irfeature, axis=1)
df.shape

# What relevant features might be missing, but can be computed?
# Race color: can be computed from rater1 and rater2
# age: can be computed from birthday
# df['age'] = (pd.to_datetime('01.07.2014') - pd.to_datetime(df['birthday'], errors='coerce')).astype('<m8[Y]') 
df['skinCol']=(df['rater1'] + df['rater2']) / 2
df['semiredCards'] = df['yellowReds'] + df['redCards']

# Are there missing data values (e.g. missing skin color ratings), and how should they be dealt with?
# Yes. Censoring exists in the data (N_skincolor_na = 21407). To overcome the issue, simply removal or imputation works.
#df[['rater1','rater2']].isna().sum()
df['skinCol'].isna().sum()
df = df.dropna()
df.shape

# How good are the skin color ratings? Do the raters agree?
# The two raters disagree about player skintone quite often (N_agree = 88365, N=27092). 
# Besidesm censoring data in players photos also leads to missing color ratings.
ra = df['rater1'] == df['rater2']
ra.value_counts()

# Should referees with very few appearances be excluded from the dataset?
# No.

# Should features be normalized and/or centralized?
# Yes. 

# Disaggregater data
def Disaggregater(df):
    df['refCount']=0

    #add a column which tracks how many games each ref is involved in
    refs=pd.unique(df['refNum'].values.ravel()) #list all unique ref IDs

    #for each ref, count their dyads
    for r in refs:
        df['refCount'][df['refNum']==r]=len(df[df['refNum']==r])    

    colnames=list(df.columns)

    j = 0
    out = [0 for _ in range(sum(df['games']))]

    for _, row in df.iterrows():
            n = row['games']
            #d =  row['redCards']       # row['allreds']
            c =  row['semiredCards']   # row['allredsStrict']
            
            #row['games'] = 1        
            
            for _ in range(n):
                row['semiredCards'] = 1 if (c-_) > 0 else 0
                #row['redCards'] = 1 if (d-_) > 0 else 0
                rowlist=list(row)  #convert from pandas Series to prevent overwriting previous values of out[j]
                out[j] = rowlist
                j += 1
                if j%1000==0:    
                    print("Number " + str(j) + " of " + str(df.shape[0]))

    dfout = pd.DataFrame(out, columns=colnames)
    
    return dfout

df1 = Disaggregater(df)
df1.shape
df1.head()

# Clean-up data
cleanVar = ['yellowReds','redCards','rater1','rater2']
df2 = df1.drop(cleanVar, axis=1)
# plot



# Finally, One-hot encode categorial variables
def OneHotEncoder(df, Var, combined):
    for v in Var:
        ohe = pd.get_dummies(df[v],v)

        if combined == True:
            df = df.drop([v], axis=1)
            df[v]= ohe.values.tolist()

        df = df.drop([v], axis=1)
        df = pd.concat([df, ohe], axis=1)

    return df

cateVar = ['club', 'leagueCountry', 'position', 'refCountry']
df2 = OneHotEncoder(df1, Var=cateVar, combined=False)




# 2.2 Model Creation (8 points)





###################################################################

df
df2['redcard']
np.unique(df2['redCards'])
np.unique(df['playerShort'])
len(np.unique(df2['playerShort']))
len(np.unique(df['playerShort']))
#tmp = df.iloc[1:10,:]

df.columns.values
other = [e for e in df.columns.values if e not in ['games','semiredCards','refNum','playerShort']]









df['refCountry']

np.unique(df['refCountry'])

position = pd.get_dummies(df['position'],'position')
column = list(position.columns.values)
position['combined']= position.values.tolist()
position[column].astype(str).sum(axis=1)
