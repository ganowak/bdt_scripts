from pickle import load
import pandas as pd
import  matplotlib.pyplot as plt
from tabulate import tabulate
# plt.style.use('/Users/camwilson/Documents/plot_styles/custom_plot_style.txt')

from definitions import run_multiclass_bdt, run_single_bdt, significances_multiclass, significances_wwttbar
from dataframes import df_tot

import os
name_of_plots_folder = 'Plots'
os.makedirs(name_of_plots_folder, exist_ok=True)


# =====================================================================================================================


# First, running all through wwdfdy bdt, then keeping >0.36 and running through wwttbar bdt


# =====================================================================================================================

print('\nRUNNING WW,DFDY,TTBAR THROUGH WWDFDY THEN WWTTBAR BDT...')

with open('wwdfdy_trained_bdt.pkl', "rb") as f:
    imported_bdt = load(f)
    

mask = (df_tot['Process']=='ww') | (df_tot['Process']=='dfdy') | (df_tot['Process']=='ttbar')
dataframe = df_tot[mask]

results_wwdfdy = run_single_bdt(imported_bdt["Model"], dataframe, imported_bdt["Features"])

ww = results_wwdfdy[results_wwdfdy['Process']=='ww']
dfdy = results_wwdfdy[results_wwdfdy['Process']=='dfdy']
ttbar = results_wwdfdy[results_wwdfdy['Process']=='ttbar']

passed_ww = ww[ww['ww_prob']>0.36]
passed_dfdy = dfdy[dfdy['ww_prob']>0.36]
passed_ttbar = ttbar[ttbar['ww_prob']>0.36]

ww_efficiency = len(passed_ww)/len(ww)
dfdy_efficiency = len(passed_dfdy)/len(dfdy)
ttbar_efficiency = len(passed_ttbar)/len(ttbar)

# print('\nEfficiencies from wwdfdy BDT (> 0.36)\n'+'='*len('Efficiencies from wwdfdy BDT (> 0.36)'),
#       f'\nww: {ww_efficiency:.4f}\ndfdy: {dfdy_efficiency:.4f}\nttbar: {ttbar_efficiency:.4f}\n'+'='*len('Efficiencies from wwdfdy BDT (> 0.36)'))

ww_efficiency_wwdfdy   = ww_efficiency
dfdy_efficiency_wwdfdy = dfdy_efficiency
ttbar_efficiency_wwdfdy = ttbar_efficiency

plt.figure(figsize=(10,8))
plt.hist(ww['ww_prob'], bins=100, histtype='step', label='ww', density=True, linewidth=2)
plt.hist(ttbar['ww_prob'], bins=100, histtype='step', label='ttbar', density=True, linewidth=2)
plt.hist(dfdy['ww_prob'], bins=100, histtype='step', label='dfdy', density=True, linewidth=2)
plt.axvline(0.36, linestyle='--', color='k', linewidth=2, label='best cut (0.36)')
plt.xlabel('WW Probability')
plt.ylabel('Normalized Counts')
plt.legend()
plt.tight_layout()
plt.savefig(name_of_plots_folder+'/wwttbardfdy_initial_scores.png', dpi=300)

new_dataframe = pd.concat([passed_ww, passed_dfdy, passed_ttbar])

with open('wwttbar_trained_bdt.pkl', "rb") as f:
    imported_bdt = load(f)
    
results_ttbar = run_single_bdt(imported_bdt["Model"], new_dataframe, imported_bdt["Features"])
best_cut, _, _, _ = significances_multiclass(results_ttbar)

new_ww = results_ttbar[results_ttbar['Process']=='ww']
new_dfdy = results_ttbar[results_ttbar['Process']=='dfdy']
new_ttbar = results_ttbar[results_ttbar['Process']=='ttbar']

new_passed_ww = new_ww[new_ww['ww_prob']>best_cut]
new_passed_dfdy = new_dfdy[new_dfdy['ww_prob']>best_cut]
new_passed_ttbar = new_ttbar[new_ttbar['ww_prob']>best_cut]

new_ww_efficiency = len(new_passed_ww)/len(ww)
new_dfdy_efficiency = len(new_passed_dfdy)/len(dfdy)
new_ttbar_efficiency = len(new_passed_ttbar)/len(ttbar)

# print(f'\nEfficiencies from wwttbar BDT (> {best_cut})\n'+'='*len('Efficiencies from wwttbar BDT (> 0.36)'),
#       f'\nww: {new_ww_efficiency:.4f}\ndfdy: {new_dfdy_efficiency:.4f}\nttbar: {new_ttbar_efficiency:.4f}\n'+'='*len('Efficiencies from wwttbar BDT (> 0.36)'))

plt.figure(figsize=(10,8))
plt.hist(new_ww['ww_prob'], bins=100, histtype='step', label='ww', density=True, linewidth=2)
plt.hist(new_ttbar['ww_prob'], bins=100, histtype='step', label='ttbar', density=True, linewidth=2)
plt.hist(new_dfdy['ww_prob'], bins=100, histtype='step', label='dfdy', density=True, linewidth=2)
plt.axvline(best_cut, linestyle='--', color='k', linewidth=2, label=f'best cut ({best_cut})')
plt.xlabel('WW Probability')
plt.ylabel('Normalized Counts')
plt.legend()
plt.tight_layout()
plt.savefig(name_of_plots_folder+'/wwttbardfdy_final_scores.png', dpi=300)

# =====================================================================================================================


# Now, running just wwttbar through a wwttbar bdt


# =====================================================================================================================

print('\n\nRUNNING WWTTBAR THROUGH WWTTBAR BDT ONLY...')

with open('wwttbar_trained_bdt.pkl', "rb") as f:
    imported_bdt = load(f)

mask = (df_tot['Process']=='ww') | (df_tot['Process']=='ttbar')
dataframe = df_tot[mask]

results = run_single_bdt(imported_bdt["Model"], dataframe, imported_bdt["Features"])

ww = results[results['Process']=='ww']
ttbar = results[results['Process']=='ttbar']

wwttbar_cut, _, _, _ = significances_wwttbar(results)

passed_ww = ww[ww['ww_prob']>wwttbar_cut]
passed_ttbar = ttbar[ttbar['ww_prob']>wwttbar_cut]

ww_efficiency = len(passed_ww)/len(ww)
ttbar_efficiency = len(passed_ttbar)/len(ttbar)

ww_efficiency_wwttbar   = ww_efficiency
ttbar_efficiency_wwttbar = ttbar_efficiency

# print(f'\nEfficiencies from wwttbar BDT (> {wwttbar_cut})\n'+'='*len('Efficiencies from wwttbar BDT (> 0.36)'),
#       f'\nww: {ww_efficiency:.4f}\nttbar: {ttbar_efficiency:.4f}\n'+'='*len('Efficiencies from wwttbar BDT (> 0.36)'))

plt.figure(figsize=(10,8))
plt.hist(ww['ww_prob'], bins=100, histtype='step', label='ww', density=True, linewidth=2)
plt.hist(ttbar['ww_prob'], bins=100, histtype='step', label='ttbar', density=True, linewidth=2)
plt.axvline(wwttbar_cut, linestyle='--', color='k', linewidth=2, label=f'best cut ({wwttbar_cut})')
plt.xlabel('WW Probability')
plt.ylabel('Normalized Counts')
plt.legend()
plt.tight_layout()
plt.savefig(name_of_plots_folder+'/wwttbar_scores.png', dpi=300)


# =====================================================================================================================


# Now, doing the multiselection bdt


# =====================================================================================================================

print('\n\nRUNNING THROUGH MULTICLASS BDT...')

with open('multiclass_bdt.pkl', "rb") as f:
    imported_bdt = load(f)
    
mask = (df_tot['Process']=='ww') | (df_tot['Process']=='ttbar') | (df_tot['Process']=='dfdy')
dataframe = df_tot[mask]

multi_results = run_multiclass_bdt(imported_bdt["Model"], dataframe, imported_bdt["Features"])

ww = multi_results[multi_results['Process']=='ww']
dfdy = multi_results[multi_results['Process']=='dfdy']
ttbar = multi_results[multi_results['Process']=='ttbar']

multi_cut, _, _, _ = significances_multiclass(multi_results)

passed_ww = ww[ww['ww_prob']>multi_cut]
passed_dfdy = dfdy[dfdy['ww_prob']>multi_cut]
passed_ttbar = ttbar[ttbar['ww_prob']>multi_cut]

ww_efficiency = len(passed_ww)/len(ww)
dfdy_efficiency = len(passed_dfdy)/len(dfdy)
ttbar_efficiency = passed_ttbar['Weight'].sum()/ttbar['Weight'].sum()
ttbar_efficiency_og = len(passed_ttbar)/len(ttbar)

# print(f'\nEfficiencies from wwdfdy BDT (> {multi_cut})\n'+'='*len('Efficiencies from wwdfdy BDT (> 0.36)'),
#       f'\nww: {ww_efficiency:.4f}\ndfdy: {dfdy_efficiency:.4f}\nttbar: {ttbar_efficiency:.4f}\n'+'='*len('Efficiencies from wwdfdy BDT (> 0.36)'))

plt.figure(figsize=(10,8))
plt.hist(ww['ww_prob'], bins=100, histtype='step', label='ww', density=True, linewidth=2)
plt.hist(ttbar['ww_prob'], bins=100, histtype='step', label='ttbar', density=True, linewidth=2)
plt.hist(dfdy['ww_prob'], bins=100, histtype='step', label='dfdy', density=True, linewidth=2)
plt.axvline(multi_cut, linestyle='--', color='k', linewidth=2, label=f'best cut ({multi_cut})')
plt.xlabel('WW Probability')
plt.ylabel('Normalized Counts')
plt.legend()
plt.tight_layout()
plt.savefig(name_of_plots_folder+'/multiclass_scores.png', dpi=300)


table_data = [
    ["ww", f"{ww_efficiency_wwdfdy:.4f}", f"{new_ww_efficiency:.4f}", f"{ww_efficiency_wwttbar:.4f}", f"{ww_efficiency:.4f}"],
    ["dfdy", f"{dfdy_efficiency_wwdfdy:.4f}", f"{new_dfdy_efficiency:.4f}", "N/A", f"{dfdy_efficiency:.4f}"],
    ["ttbar", f"{ttbar_efficiency_wwdfdy:.4f}", f"{new_ttbar_efficiency:.4f}", f"{ttbar_efficiency_wwttbar:.4f}", f"{ttbar_efficiency:.4f}"],
]

headers = ["Process",
           "wwdfdy BDT\n(cut > 0.36)",
           f"wwdfdy \u2192 wwttbar BDT\n(cut > {best_cut})",
           f"wwttbar BDT only\n(cut > {wwttbar_cut})",
           f"Multiclass BDT\n(cut > {multi_cut})"]

print('\n\nEFFICIENCY SUMMARY')
print(tabulate(table_data, headers=headers, tablefmt='fancy_grid', stralign='center', numalign='center'))
