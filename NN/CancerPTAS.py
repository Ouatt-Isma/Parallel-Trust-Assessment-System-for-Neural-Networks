import sys

# Specify the path to the folder containing the file
import os 
folder_path = f"{os.getcwd()}/"
sys.path.append(folder_path)


import socket
import pickle
from PTASTemp.ptasInterface import PTASInterface
from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
import numpy as np
from concrete.TrustOpinion import TrustOpinion 
from matplotlib import pyplot as plt

from concrete.ArrayTO import ArrayTO
import time 
from PTAStemplate import PTAS 
from utils import writeto


input_dim = 30
hidden_dim = 16
output_dim = 2

def T_NOT_poisoned(x: np.array, dim):
    op = "trust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T_poisoned(x: np.array, dim):
    op = "trust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T1(x: np.array, dim):
    op = "vacuous" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
def Ttrust(x: np.array, dim):
    op = "trust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def Tdistrust(x: np.array, dim):
    op = "distrust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def Tvacuous(x: np.array, dim):
    op = "vacuous" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T2(x: np.array, dim):

    op = "vacuous" 
    n = len(x)
    res = np.empty((n, dim))
    for i in range(n):
        for j in range(dim):
            res[i][j] = TrustOpinion.ftrust()

def Tgen(xx, yy):
    assert xx[0]=='x'
    assert yy[0]=='y'
    def inner_function(x: np.array, dim):
        if(dim==input_dim):
            op = xx[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
        if(dim==output_dim):
            op = yy[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
    return inner_function






# XX = ["xvacuous2"]
# XX = ['xvacuous']

# YY = ['yvacuous2']
# YY = ['yvacuous']

XX = ["xtrust",  "xdistrust", "xvacuous"]
YY = ["ytrust", "ydistrust", "yvacuous"]



def run_uni_test(xx, yy, epsilon_low, epsilon_up):
    assert xx in XX
    assert yy in YY
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    Tf = Tgen(xx, yy)
    # ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, structure = [input_dim, hidden_dim, output_dim], epsilon_low=10**(-3), epsilon_up=10**(-2))
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, 
                structure = [input_dim, hidden_dim, output_dim], 
                epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True)
    
    datapath = folder_path+'res/'+xx+yy+'/'
    try:
        os.mkdir(datapath)
    except:
        pass
    ptas.run_chunk()

    PTAS.eval_plot(ptas.EVAL, output_dim, None,f'{datapath}all.pdf', n_epoch=15)
    # PTAS.eval_plot_simpl(ptas.EVAL, output_dim, None,f'{datapath}simpl.pdf')
    # PTAS.eval_plot_aggr(ptas.EVAL, output_dim, None,f'{datapath}aggr.pdf')
    # print("--------------------------- 0 0 0 ----------------------")
    # print(ptas.omega_thetas[0].get_shape())
    # print(ptas.omega_thetas[0])
    # print("--------------------------- 1 1 1 ----------------------")
    # print(ptas.omega_thetas[1].get_shape())
    # print(ptas.omega_thetas[1])
    # print("-------------------------------------------------")
    
    # writeto(ptas.omega_thetas, datapath+"\\omegas.pkl")
    # at = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="trust")))
    # # print(at)
    # writeto(at, datapath+"\\at.pkl")
    # av = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    # # print(av)
    # writeto(av, datapath+"\\av.pkl")
    # ad = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))
    # # print(ad)
    # writeto(ad, datapath+"\\ad.pkl")
    

def test():
    epsilon_low = 0.01
    epsilon_up = None
    for xx in XX:
        for yy in YY:
            print(f"Running Test for {xx}, {yy}")
            print()
            run_uni_test(xx, yy, epsilon_low, epsilon_up)
def main_cancer():
    input_dim = 30
    hidden_dim = 16
    output_dim = 2
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]
    
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tdistrust, structure = [input_dim, hidden_dim, output_dim], epsilon_low=0.03, eval=True)
    datapath = folder_path+'NN\\eval_cancer'
    try:
        os.mkdir(datapath)
    except:
        pass

    ptas.run_chunk()
    print("--------------------------- 0 0 0 ----------------------")
    print(ptas.omega_thetas[0].get_shape())
    print(ptas.omega_thetas[0])
    print("-------------------------------------------------")
    print()

    print("--------------------------- 1 1 1 ----------------------")
    print(ptas.omega_thetas[1].get_shape())
    print(ptas.omega_thetas[1])
    print("-------------------------------------------------")
    print()

    print("Apply Feed Forward on fully Trusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="trust")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\at.pkl")

    print("Apply Feed Forward on Vacuous Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\av.pkl")

    print("Apply Feed Forward on fully Untrusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\ad.pkl")


def tryy():
    xx = "xtrust"
    yy = "ytrust"
    datapath = folder_path+'NN\\eval\\'+xx+yy
    with open(datapath+"\\omegas.pkl", "rb") as file:  # Use 'rb' for reading binary
        loaded_data = pickle.load(file)
    print("Loaded data:", loaded_data[0])


if __name__ == "__main__":
    test()
    # simple_test()
   
    # a = TrustOpinion.ftrust()
    # c = TrustOpinion.dtrust()
    # b = TrustOpinion.vacuous()
    # t = [a,b,c]
    # for j in range(3):
    #     for i in range(3):
    #         print(f"{j}*{i} = {t[i]*t[j]}")
