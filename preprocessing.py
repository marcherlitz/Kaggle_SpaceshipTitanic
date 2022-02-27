import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from preprocessing import preprocessing

import re
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def cryoProcessing(dataDf):
    surnameList = []
    for i in range(len(dataDf['Name'])):
        fullName = dataDf['Name'].iloc[i]
        if fullName != fullName:
            surnameList.append('unknown')
        else:
            for j in range(len(fullName)):
                if fullName[j] == " ":
                    surnameList.append(fullName[j+1:])
                    
    dataDf['surname'] = surnameList
    dataDf = dataDf.drop('Name', axis=1)
    
    
    
    
    groupList = []
    for i in range(len(dataDf['PassengerId'])):
        fullId = dataDf['PassengerId'].iloc[i]
        if fullId != fullId:
            groupList.append('PassengerId')
        else:
            for j in range(len(fullId)):
                if fullId[j] == "_":
                    groupList.append(fullId[:j])
    dataDf['groups'] = groupList

    dataDf = dataDf.drop('PassengerId', axis=1)
    
    
    
    numList = []
    deckList = []
    sideList = []

    for i in range(len(dataDf['Cabin'])):
        fullCabinId = dataDf['Cabin'].iloc[i]
        if fullCabinId != fullCabinId:
            numList.append('unknown')
            deckList.append('unknown')
            sideList.append('unknown')
        else:
            num = re.findall('\d+', fullCabinId)
            numList.append(num[0])
        
            deck = fullCabinId[0]
            deckList.append(deck)
        
            side = fullCabinId[-1]
            sideList.append(side)

    dataDf['num'] = numList
    dataDf['deck'] = deckList
    dataDf['side'] = sideList

    dataDf = dataDf.drop('Cabin', axis=1)
    
    
    
    floatValueNames = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for i in floatValueNames:
        meanValue = dataDf[i].mean()
        dataDf[i] = dataDf[i].fillna(meanValue)
    #print(dataDf.columns[dataDf.isna().any()].tolist())
    
    
    sum_column = dataDf["RoomService"] + dataDf["FoodCourt"] + dataDf["ShoppingMall"] + dataDf["Spa"] + dataDf["VRDeck"]
    dataDf["totalBill"] = sum_column
    
    return dataDf
    

def preprocessing_code(dataDf):
    surnameList = []
    for i in range(len(dataDf['Name'])):
        fullName = dataDf['Name'].iloc[i]
        if fullName != fullName:
            surnameList.append('unknown')
        else:
            for j in range(len(fullName)):
                if fullName[j] == " ":
                    surnameList.append(fullName[j+1:])
                    
    dataDf['surname'] = surnameList
    dataDf = dataDf.drop('Name', axis=1)
    
    
    
    
    groupList = []
    for i in range(len(dataDf['PassengerId'])):
        fullId = dataDf['PassengerId'].iloc[i]
        if fullId != fullId:
            groupList.append('PassengerId')
        else:
            for j in range(len(fullId)):
                if fullId[j] == "_":
                    groupList.append(fullId[:j])
    dataDf['groups'] = groupList

    dataDf = dataDf.drop('PassengerId', axis=1)
    
    
    
    numList = []
    deckList = []
    sideList = []

    for i in range(len(dataDf['Cabin'])):
        fullCabinId = dataDf['Cabin'].iloc[i]
        if fullCabinId != fullCabinId:
            numList.append('unknown')
            deckList.append('unknown')
            sideList.append('unknown')
        else:
            num = re.findall('\d+', fullCabinId)
            numList.append(num[0])
        
            deck = fullCabinId[0]
            deckList.append(deck)
        
            side = fullCabinId[-1]
            sideList.append(side)

    dataDf['num'] = numList
    dataDf['deck'] = deckList
    dataDf['side'] = sideList

    dataDf = dataDf.drop('Cabin', axis=1)
    
    
    
    floatValueNames = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for i in floatValueNames:
        meanValue = dataDf[i].mean()
        dataDf[i] = dataDf[i].fillna(meanValue)
    #print(dataDf.columns[dataDf.isna().any()].tolist())
    
    
    dataDf = dataDf.fillna('unknown')
    
    sum_column = dataDf["RoomService"] + dataDf["FoodCourt"] + dataDf["ShoppingMall"] + dataDf["Spa"] + dataDf["VRDeck"]
    dataDf["totalBill"] = sum_column
    
    dataDf['VIP'] = dataDf['VIP'].astype(str)
    dataDf['CryoSleep'] = dataDf['CryoSleep'].astype(str)
    #dataDf['Transported'] = dataDf['Transported'].astype(str)
    
    tmpDataDf = dataDf[['HomePlanet', 'Destination', 'surname', 'num', 'deck', 'side', 'groups', 'VIP', 'CryoSleep']]

    dataDf[['HomePlanet', 'Destination', 'surname', 'num', 'deck', 'side', 'groups', 'VIP', 'CryoSleep']] = tmpDataDf.apply(LabelEncoder().fit_transform)

    #dataDf = dataDf.apply(LabelEncoder().fit_transform)
    return dataDf
    
    