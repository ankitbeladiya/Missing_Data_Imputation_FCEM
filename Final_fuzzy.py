"""
Created on Sat May 21 17:39:21 2017

@author: ankit
"""

import pandas as pd
import numpy as np
import os
import glob
from scipy.spatial.distance import cdist


class Encryption(object):

    def __init__(self):
        self.unique_value_ = []
        self.numeric_ = []
        self.DataType = []

    def do_encraption(self, data):
        self.DataType = pd.Series(data.dtypes)
        self.DataType = self.DataType.index[self.DataType == 'O']
        for i in self.DataType:
            p = data[i].dropna().unique()
            for l in range(len(p)):
                self.unique_value_.append(p[l])
        self.unique_value_ = set(self.unique_value_)
        self.numeric_ = np.arange(1, len(self.unique_value_)+1, 1)
        data.replace(to_replace=self.unique_value_, value=self.numeric_, inplace=True)
        return data

    def do_decription(self, data):
        for i in self.DataType:
            data[i] = data[i].apply(round)
            data[i].replace(to_replace=self.numeric_, value=self.unique_value_, inplace=True)        
        return data

def _cmeans0(data, u_old, c, m):
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d


def _distance(data, centers):
    return cdist(data, centers).T


def _fp_coeff(u):
    n = u.shape[1]
    return np.trace(u.dot(u.T)) / float(n)


def initu0(data, c, init, seed):
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)
    return u


def cmeans(data, c, m, error, maxiter, init=None, seed=None):
    u = initu0(data,c,init, seed)

    jm = np.zeros(0)
    p = 0

    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr



def cmeans_predict(test_data, cntr_trained, m, error, maxiter, init=None,seed=None):
    c = cntr_trained.shape[0]

    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [u, Jjm, d] = _cmeans_predict0(test_data, cntr_trained, u2, c, m)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)
    return u



def _cmeans_predict0(test_data, cntr, u_old, c, m):
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    d = _distance(test_data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return u, jm, d


def fitfuzzy(data, ncenters,m):
    print('--Fittin Model--')
    data = data.T
    cntr = cmeans(data, ncenters, m, error=0.005, maxiter=100, init=None)
    return cntr



def fuzzEM(misdata, centr, u, m):
    misdata[misdata[misdata.isnull()].index[0]] = np.dot(u[:, misdata.name]**m, centr[:, misdata[misdata.isnull()].index[0]])/np.sum(u[:, misdata.name]**m)
    return misdata

def imputation(data, c, m,error , max_itr):
    centr = fitfuzzy(data.dropna(axis=0), c ,m)
    u= cmeans_predict(data.fillna(value=0).T, centr, m, error, max_itr)
    completeData =  data.apply(lambda row :fuzzEM(row, centr, u ,m) if pd.isna(row).any() else row,axis =1)
    return completeData


def cal_nrms(a, b):
    print('--Calculating NRMS Value--')
    return np.sqrt(np.square(np.subtract(a, b)).sum().sum())/np.sqrt(np.square(a).sum().sum())


def write_nrms(a, b):
    print('--Generating NRMS.xlsx File--')
    nrms = pd.DataFrame(data=[a, b], index=['Data_set', 'NRMS_value'])
    nrms = nrms.T
    nrms['Data_set'] = nrms['Data_set'].str.replace('.xlsx', '')
    writer = pd.ExcelWriter('NRMS.xlsx')
    nrms.to_excel(writer, index=False, header=None)
    writer.save()
    print('--Done-- ')
    return


def main():
    DatasetName = '/HOV'  # data set to run
    IncompleteDatasetPath = '/home/ankit/GitProject/Missing_Data_Imputation/Incomplete_data'+DatasetName  # provide path for incompelet data
    ImputedDatasetPath = '/home/ankit/GitProject/Missing_Data_Imputation/Imputed_data'+DatasetName     # provide path to write imputed data excel sheet
    CompleteDatasetPath = '/home/ankit/GitProject/Missing_Data_Imputation/Complete_data'+DatasetName+'.xlsx'   # provide path for complete data  
    NoOfCenters = 10  # No of clusters in data

    CompleteData = pd.read_excel(CompleteDatasetPath, header=None)

    EncryptionFlag = object in list(CompleteData.dtypes)  # flag for encryption

    if EncryptionFlag:
        EncriptData = Encryption()
        CompleteData = EncriptData.do_encraption(CompleteData)

    os.chdir(IncompleteDatasetPath)
    files = glob.glob('*.xlsx')

    nrms = []
    for f in files:
        print('-----------Working on '+f+'-----------')
        os.chdir(IncompleteDatasetPath)
        IncompleteData = pd.read_excel(f, header=None)

        if EncryptionFlag:
            print('--Doing Encryption--')
            EncriptIncompleteData = Encryption()
            IncompleteData = EncriptIncompleteData.do_encraption(IncompleteData)

        print('--Imputing Missing Value--')
        ImputedData = imputation(IncompleteData, NoOfCenters, m = 2,error=0.001,max_itr = 100)
        NrmsValue = cal_nrms(CompleteData, ImputedData)

        nrms.append(NrmsValue)
        print('--NRMS value for '+f+' : {}'.format(NrmsValue))

        if EncryptionFlag:
            print('--Doing decryption--')
            ImputedData = EncriptIncompleteData.do_decription(ImputedData)
        os.chdir(ImputedDatasetPath)
        writer = pd.ExcelWriter(f)
        ImputedData.to_excel(writer, index=False, header=None)
        writer.save()

        print('--done--')
        print('\n')
        print('\n')
    os.chdir(ImputedDatasetPath)
    write_nrms(files, nrms)


if __name__ == '__main__':
    main()
