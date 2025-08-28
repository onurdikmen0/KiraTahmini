"""
Veri Bilimcinin El Çantası 
"""

from unidecode import unidecode
import pandas as pd
import numpy as np



def f(df):
    """
    Fonksiyon : Dataframedeki tüm kolonların küçük karakterle, boşluklara "_" koyar ve tükçe karakterleri ingilizce karakterler ile değiştirir.
    """
    return df.columns.map(lambda x: unidecode(x).replace(" ", "_").lower())


def selam_ver():
    print("hola")


def bos_veri_orani(df):
    """
    Fonksiyon : 
        Veri setindeki kolonların boş verilerinin oranını geri döndürür.
    Parametre :
        df : dataframe
    """
    return round((df.isnull().sum() / df.shape[0])*100, 2)


def outliers(df,col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]
   