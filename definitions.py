import numpy as np
import matplotlib.pyplot as plt

path='./Data/'
# plt.style.use('/Users/camwilson/Documents/plot_styles/custom_plot_style.txt')


def run_single_bdt(classifier, dataframe, feature_names):

    x_input = dataframe[feature_names]
    
    probabilities = classifier.predict_proba(x_input)
    bdt_scores = probabilities[:, 1]

    results = dataframe.copy()
    results['ww_prob'] = bdt_scores
    
    return results

def run_multiclass_bdt(classifier, dataframe, feature_names):

    x_input = dataframe[feature_names]
    probabilities = classifier.predict_proba(x_input)
    results = dataframe.copy()

    # traditional bdt score is prob[:,0], the prob ratio of an event being WW
    results['ww_prob'] = probabilities[:,0]
    results['dfdy_prob'] = probabilities[:,1]
    results['ttbar_prob'] = probabilities[:,2]
    
    return results

def significances_wwdfdy(dataframe, step=0.01):
    
    # Defining how many thresholds there should be in the bdt scores
    list_of_thresholds = np.arange(0, 1.0, step)
    significances = []
    
    dataframe = dataframe.copy()
    dataframe.loc[dataframe['Process'] == 'ww', 'Signal'] = 1
    dataframe.loc[dataframe['Process'] == 'dfdy', 'Signal'] = 0

    # Iterating through each threshold
    for cut in list_of_thresholds:
        # Defining events which pass this threshold iteration
        passed = dataframe[dataframe['ww_prob'] > cut]

        # Summing up the weights of pass/fail this threshold iteration
        s = passed[passed['Signal'] == 1]['Weight'].sum()
        b = passed[passed['Signal'] == 0]['Weight'].sum()

        # Significance formula (with safety if dividing by zero)
        significance = s / np.sqrt(s + b) if (s + b) > 0 else 0
        significances.append(significance)

    # Take the best significance out of all the significances and get that threshold value
    max_significance = max(significances)
    best_cut = list_of_thresholds[np.argmax(significances)]

    return best_cut, max_significance, list_of_thresholds, significances

def significances_wwttbar(dataframe, step=0.01):
    
    # Defining how many thresholds there should be in the bdt scores
    list_of_thresholds = np.arange(0, 1.0, step)
    significances = []
    
    dataframe = dataframe.copy()
    dataframe.loc[dataframe['Process'] == 'ww', 'Signal'] = 1
    dataframe.loc[dataframe['Process'] == 'ttbar', 'Signal'] = 0

    # Iterating through each threshold
    for cut in list_of_thresholds:
        # Defining events which pass this threshold iteration
        passed = dataframe[dataframe['ww_prob'] > cut]

        # Summing up the weights of pass/fail this threshold iteration
        s = passed[passed['Signal'] == 1]['Weight'].sum()
        b = passed[passed['Signal'] == 0]['Weight'].sum()

        # Significance formula (with safety if dividing by zero)
        significance = s / np.sqrt(s + b) if (s + b) > 0 else 0
        significances.append(significance)

    # Take the best significance out of all the significances and get that threshold value
    max_significance = max(significances)
    best_cut = list_of_thresholds[np.argmax(significances)]

    return best_cut, max_significance, list_of_thresholds, significances

def significances_multiclass(dataframe, step=0.01):
    
    # Defining how many thresholds there should be in the bdt scores
    list_of_thresholds = np.arange(0, 1.0, step)
    significances = []
    
    dataframe = dataframe.copy()
    dataframe.loc[dataframe['Process'] == 'ww', 'Signal'] = 0
    dataframe.loc[dataframe['Process'] == 'dfdy', 'Signal'] = 1
    dataframe.loc[dataframe['Process'] == 'ttbar', 'Signal'] = 2

    # Iterating through each threshold
    for cut in list_of_thresholds:
        # Defining events which pass this threshold iteration
        passed = dataframe[dataframe['ww_prob'] > cut]

        # Summing up the weights of pass/fail this threshold iteration
        s = passed[passed['Signal'] == 0]['Weight'].sum()
        b = passed[passed['Signal'] == 1]['Weight'].sum() + passed[passed['Signal'] == 2]['Weight'].sum()

        # Significance formula (with safety if dividing by zero)
        significance = s / np.sqrt(s + b) if (s + b) > 0 else 0
        significances.append(significance)

    # Take the best significance out of all the significances and get that threshold value
    max_significance = max(significances)
    best_cut = list_of_thresholds[np.argmax(significances)]

    return best_cut, max_significance, list_of_thresholds, significances

