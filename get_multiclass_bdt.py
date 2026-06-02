from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

from dataframes import df_tot

path = './Data/'

mask = (df_tot['Process']=='ww') | (df_tot['Process']=='dfdy') | (df_tot['Process']=='ttbar')
dataframe = df_tot[mask]

def get_classifier(dataframe):

    dataframe = dataframe.copy()
    dataframe.loc[dataframe['Process'] == 'ww', 'Signal'] = 0
    dataframe.loc[dataframe['Process'] == 'dfdy', 'Signal'] = 1
    dataframe.loc[dataframe['Process'] == 'ttbar', 'Signal'] = 2

    features = dataframe.drop(['Signal', 'Weight', 'Process'], axis=1)
    feature_names = features.columns.tolist()
    
    target = dataframe['Signal']
    weights = dataframe['Weight']
    
    x_train, x_test, y_train, y_test, weight_train, weight_test = train_test_split(features, target, weights, train_size=0.8)
    bdt = HistGradientBoostingClassifier()
    classifier = bdt.fit(x_train, y_train, sample_weight=weight_train)
    test_dataframe = dataframe.loc[x_test.index].copy()

    return classifier, feature_names, [x_train, x_test], [y_train, y_test], test_dataframe

trained_model, feature_names, trained_on, tested_on, test_dataframe = get_classifier(dataframe)
trained_bdt_info = {
    'Model': trained_model,
    'Features': feature_names,
    'Trained On': trained_on,
    'Tested On': tested_on,
    'Test Dataframe': test_dataframe
}

from pickle import dump
with open("multiclass_bdt.pkl", "wb") as f:
    dump(trained_bdt_info, f, protocol=5)


print('\nBDT Exported as pickle successfully!')
