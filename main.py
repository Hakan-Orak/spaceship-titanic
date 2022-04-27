import numpy
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def init():
    global df, test_ID, train_ID

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train.info(), test.info()

    test_ID = test['PassengerId']
    train_ID = train['PassengerId']
    df = pd.concat([train, test], axis=0)


def normalizeVar():
    df['HomePlanet'].replace('Earth', 0, inplace=True)
    df['HomePlanet'].replace('Mars', 1, inplace=True)
    df['HomePlanet'].replace('Europa', 2, inplace=True)

    df['CryoSleep'].replace(False, 0, inplace=True)
    df['CryoSleep'].replace(True, 1, inplace=True)


def NormalizeData():
    passenger_cols = ('HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name', 'Transported')

    df[passenger_cols] = df[passenger_cols].apply(LabelEncoder().fit_transform())
    df.head()

def LabelEncoderData():
    passenger = ['CryoSleep','VIP']
    df[passenger] = df[passenger].apply(LabelEncoder().fit_transform)
    print(df['VIP'].head(10))

def OneHotEncoderData():
    df_transformed = pd.get_dummies(df,columns=['HomePlanet', 'Destination'])

    pd.set_option('display.max_columns', None)
    print(df_transformed.head(10))

def visualizeData():
    init()

    df.info(), print(df.head())

    print(" champ manquant : ")
    print(df.isnull().sum())

    print(df['HomePlanet'].value_counts())
    print(df['HomePlanet'].isnull().sum())
    print(df['CryoSleep'].value_counts())
    print(df['Cabin'].isnull().sum())

    print("\n\n variable aprés traitement")



    print(df['HomePlanet'].value_counts())
    print(df['CryoSleep'].value_counts())
    print('\n Label Encoder :')
    LabelEncoderData()
    print('\n OneHotEncoder :')
    OneHotEncoderData()


def standardScalerVariablesNumeriques():
    standardScalerData = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    dfStandardScalerData = df[standardScalerData]

    for i in standardScalerData:
        df[i] = df[i].fillna(0).astype(int)

    print("dfStandardScalerData")
    print(dfStandardScalerData)

    scale = StandardScaler().fit_transform(dfStandardScalerData)
    print(scale)

# visualizeData()
# NormalizeData()
init()
standardScalerVariablesNumeriques()
