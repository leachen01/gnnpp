import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lightning as L

from utils.data import load_dataframes, load_distances, normalize_features_and_create_graphs, load_complete_dataframe

def main():
    # !! still haven't loaded 72h and 120h
    dataframes = load_dataframes(mode="train", leadtime="120h")
    #(train_rf, train_rf_target) = dataframes["train"]
    #(test_rf, test_rf_target) = dataframes["test_rf"]
    #(test_f, test_f_target) = dataframes["test_f"]
    stations_f = dataframes["stations"]

    # load the complete data and save it as picklefile

    hyp_dataframes = load_dataframes(mode="hyperopt", leadtime="120h")
    #(htrain_rf, htrain_rf_target) = hyp_dataframes["train"]
    #(valid_rf, valid_rf_target) = hyp_dataframes["valid"]
    #hstations_f = hyp_dataframes["stations"]

    #print(f"Train_rf: {train_rf}")
    #print(f"Valid_rf: {valid_rf}")
    #print(f"ensemble members in train: {train_rf.number.nunique()}")
    #print(f"ensemble members in valid: {valid_rf.number.nunique()}")
    #print(f"train entries: {train_rf.shape[0]}")
    #print(f"valid entries: {valid_rf.shape[0]}")
    #print(f"divided by ensemble members in train per station (1997-2009): {train_rf.shape[0]/train_rf.number.nunique()/122}")
    #print(f"divided by ensemble members in valid per station (2010-2013): {valid_rf.shape[0] / valid_rf.number.nunique() / 122}")
    #print(f"sum (1997-2013): {(train_rf.shape[0]/train_rf.number.nunique()/122) + (valid_rf.shape[0] / valid_rf.number.nunique() / 122)}")
    # print(f"divided by ensemble members in f per station (2017-1018): {test_f.shape[0]/test_f.number.nunique()/122}")

    #df = load_complete_dataframe()

    #print(df)

    dataframes = load_dataframes(mode="eval", leadtime="120h")
    #(train_rf, train_rf_target) = dataframes["train"]
    #(test_rf, test_rf_target) = dataframes["test_rf"]
    #(test_f, test_f_target) = dataframes["test_f"]
    #print(f"train_rf: {train_rf}")
    #print(f"test_rf: {test_rf}")
    #print(f"divided by ensemble members in train per station (1997-2013): {train_rf.shape[0]/train_rf.number.nunique()/122}")
    #print(f"divided by ensemble members in test_rf per station (2014-2017): {test_rf.shape[0] / test_rf.number.nunique() / 122}")
    #print(f"divided by ensemble members in test_f per station (2017-2018): {test_f.shape[0] / test_f.number.nunique() / 122}")
    #dist = load_distances(stations_f)
    # print(dist)
    print(stations_f)

#    graphs_train_rf, tests = normalize_features_and_create_graphs(
#        training_data=dataframes["train"],
#        valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
#        mat=dist,
#        max_dist = 100
     #    max_dist=config.max_dist, # wie lädt man denn die config? => train_ensemble
#     )
#    graphs_test_rf, graphs_test_f = tests
#    print(graphs_test_rf)


if __name__ == '__main__':
    main()

