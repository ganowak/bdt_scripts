#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 15:29:45 2026

@author: camwilson
"""

from pickle import load
import matplotlib.pyplot as plt
import math
import numpy as np
plt.style.use('/Users/camwilson/Documents/plot_styles/custom_plot_style.txt')


import os
name_of_plots_folder = 'Plots'
os.makedirs(name_of_plots_folder, exist_ok=True)


with open('WWDFDY_trained_bdt_1.8.0.pkl', "rb") as f:
    wwdfdy_imports = load(f)
    
with open('WWttbar_trained_bdt_1.8.0.pkl', "rb") as f:
    wwttbar_imports = load(f)

variables = [r'$p_\mu^{(T)}$', r'$\eta_\mu$', r'$\phi_\mu$', r'$p_e^{(T)}$', r'$\eta_e$',
       r'$\phi_e$', r'$\Delta R_{\ell\ell}$', r'$\Delta \phi_{\ell\ell}$',
       r'$\Delta \eta_{\ell\ell}$', r'$p_{\ell\ell}^{(T)}$', r'$M_{\ell\ell}$']

    
wwttbar_df = wwttbar_imports['Dataframe']
wwttbar_df = wwttbar_df[wwttbar_df['muon_PT']<200]
wwttbar_df = wwttbar_df[wwttbar_df['electron_PT']<200]
wwttbar_df = wwttbar_df[wwttbar_df['DiLeptonpT']<200]


ww = wwttbar_df[wwttbar_df['Signal']==1]
ttbar = wwttbar_df[wwttbar_df['Signal']==0]

print(ww.drop(['Weight','Signal'], axis=1).keys())

if __name__ == '__main__':

    cols = 2
    rows = math.ceil(len(variables)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
    axes = axes.flatten()
    
    for i, (key, variable) in enumerate(zip(ww.drop(['Weight','Signal'], axis=1).keys(), variables)):
        
        num_bins = 100
        bins = np.linspace(min(wwttbar_df[key]), max(wwttbar_df[key]), num_bins + 1)
        
        ax = axes[i]
        ww_counts, ww_bins, _ = ax.hist(ww[key], bins=bins, weights=ww['Weight'],
                 color='dodgerblue', histtype='step', linewidth=2.5, label=r'$W^+W^-$', alpha=0.75)
        ttbar_counts, ttbar_bins, _ = ax.hist(ttbar[key], bins=bins, weights=ttbar['Weight'], 
                 color='red', histtype='step', linewidth=2.5, label=r'$t\overline{t}$', alpha=0.75)
        ax.set_title(f'Distribution for {variable}', pad=15, fontsize=30)
        ax.set_xlabel(f'{variable}', fontsize=20)
        ax.set_ylabel('Event Count', labelpad=10, fontsize=20)
        ax.legend(fontsize=16)
        ax.grid(True, linestyle='--')
            

    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()
        
    

    plt.tight_layout()
    plt.savefig('Plots/variable_distributions.png', dpi=300)
    
