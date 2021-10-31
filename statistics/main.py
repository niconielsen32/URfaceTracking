import numpy as np
import pandas as pd
import scipy
from statsmodels.multivariate.manova import MANOVA
import seaborn as sns

# Basic variables
dir_experiment1 = "../datasets/experiment1.csv"
dir_experiment2 = "../datasets/experiment2.csv"
dir_experiment3 = "../datasets/experiment3.csv"
dependandt = ["area_bbox", "x_displacement", "y_displacement", "x_error", "y_error", "elapsed_time", "fps"]
independant = ["x_speed", "y_speed", "method"]
type_dict = {"area_bbox" : 'float64', "x_displacement" : 'float64', "y_displacement": 'float64', "x_error" : 'float64', "y_error" : 'float64', "elapsed_time": 'float64', "fps": 'float64', "x_speed": 'float64', "y_speed": 'float64', "method": 'category'}


if __name__ == "__main__":

    ########################################################################################################################################
    # Read the datasets
    experiment1 = pd.read_csv(dir_experiment1, sep=",")
    experiment2 = pd.read_csv(dir_experiment2, sep=",")
    experiment3 = pd.read_csv(dir_experiment3, sep=",")
    exp_list = [experiment1, experiment2, experiment3]
    
    # Convert the columns to the adequate type and clean NA's
    for i in range(0,len(exp_list)):
        exp_list[i] = exp_list[i].astype(type_dict)
        exp_list[i] = exp_list[i].dropna()

        ########################################################################################################################################
        # Descriptive statistics
        exp_list[i].describe()

        ########################################################################################################################################
        # Normality check


        
