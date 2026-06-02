import uproot
import pandas as pd
import pdb

path = '../TupleSkimmer/'
kinematic_keys = ["*_PT", "*_ETA", "*_PHI", "Delta*", "DiLepton*", "Scale Factor", "Reweight", "Weight"]

trees = {'ww': uproot.open(path+"WW_MG5_NLO_wID.root:OS_MuE_Reco"),
         'dfdy':uproot.open(path+"DFDY_Pythia_LO_wID.root:OS_MuE_Reco"),
         'ttbar':uproot.open(path+"ttbar_Pythia_LO_rwgt_wID.root:OS_MuE_Reco")}
    
data = {'ww': trees['ww'].arrays(filter_name=kinematic_keys, library='pd'),
        'dfdy': trees['dfdy'].arrays(filter_name=kinematic_keys, library='pd'),
        'ttbar': trees['ttbar'].arrays(filter_name=kinematic_keys, library='pd')}

data['ww']['Process'], data['dfdy']['Process'], data['ttbar']['Process'] = 'ww', 'dfdy', 'ttbar'

for process in ["ttbar", "ww"]:
    data[process]['Weight']=data[process]['Reweight']*data[process]['Scale Factor']
    data[process]=data[process].drop(['Reweight','Scale Factor'], axis=1)

df_tot = pd.concat(
    [data['ww'], data['dfdy'], data['ttbar']], 
    ignore_index=True
).sample(frac=1).reset_index(drop=True)

