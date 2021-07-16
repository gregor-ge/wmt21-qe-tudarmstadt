import run
import copy
import yaml
import argparse
import numpy as np
import os
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def create_dataset(config):
    data = {}
    for pair in config["test"]["pairs"]:
        data[str(pair[0]) + '-' + str(pair[1])] = []
    for i, path in enumerate(config['adapter_path']):
        single_config = copy.deepcopy(config)
        single_config['adapter_path'] = path
        scores, language_pairs = run.predict(single_config, save=False)
        for predictions, pair in zip(scores, language_pairs):
            data[str(pair[0]) + '-' + str(pair[1])].append(predictions)
        print()
    np.save(os.path.join('data', 'ensemble_data_train.npy'), data)


def load_data(use_pair=True, use_train=False):
    data_dev = np.load(os.path.join('data', 'ensemble_data2.npy'), allow_pickle=True).item()
    data_test = np.load(os.path.join('data', 'ensemble_data_test2.npy'), allow_pickle=True).item()
    data_train = np.load(os.path.join('data', 'ensemble_data_train.npy'), allow_pickle=True).item()
    columns = ['pair', 'model1', 'model2', 'model3', 'model4', 'model5']
    pairs = list(data_dev.keys())
    true_scores_dev = []
    true_scores_test = []
    true_scores_train = []
    for pair in pairs:
        lang1, lang2 = pair.split('-')
        da_scores_dev = pd.read_csv(f"data/data/direct-assessments/dev/{lang1}-{lang2}-dev/dev.{lang1}{lang2}.df.short.tsv",
                                delimiter="\t", quoting=3)['z_mean']
        da_scores_test = pd.read_csv(f"data/data/direct-assessments/test/{lang1}-{lang2}/test20.{lang1}{lang2}.df.short.tsv",
                    delimiter="\t", quoting=3)['z_mean']
        true_scores_test.append(da_scores_test)
        true_scores_dev.append(da_scores_dev)
    if use_train:
        for pair in pairs:
            lang1, lang2 = pair.split('-')
            da_scores_train = pd.read_csv(f"data/data/direct-assessments/train/{lang1}-{lang2}-train/train.{lang1}{lang2}.df.short.tsv",
                        delimiter="\t", quoting=3)['z_mean']
            true_scores_train.append(da_scores_train)

    true_scores_dev = np.concatenate(true_scores_dev)
    true_scores_test = np.concatenate(true_scores_test)
    if use_train:
        true_scores_train = np.concatenate(true_scores_train)
    #plt.scatter([i for i in range(len(true_scores[:1000]))], true_scores[2000:3000])
    #print(pairs)

    def process_dict(data_dict):
        all_preds = []
        all_pairs = []
        for pair, predictions in data_dict.items():
            onehot_pair = [int(pair == p) for p in pairs]
            pair_class = [pairs.index(pair) for i in range(len(predictions[0]))]
            all_pairs.append(pair_class)
            #pair_class = pd.Series([pairs.index(pair) for i in range(len(predictions[0]))]) # TODO: Maybe norm to one?
            #df['pair'] = df['pair'].append(pair_class)
            per_pair_preds = []
            for i in range(len(predictions)):
                per_pair_preds.append(np.ravel(predictions[i]))
                #df['model{}'.format(i + 1)] = df['model{}'.format(i + 1)].append(preds)
            per_pair_preds = np.vstack(per_pair_preds)
            all_preds.append(per_pair_preds)
        all_pairs = np.concatenate(all_pairs)
        all_preds = np.hstack(all_preds)
        all_data = np.vstack((all_pairs, all_preds)).T if use_pair else all_preds.T
        return all_data
    #plt.scatter([i for i in range(len(true_scores[:1000]))], all_data[:, 1][2000:3000])
    #plt.show()
    all_data_dev = process_dict(data_dev)
    all_data_test = process_dict(data_test)
    if use_train:
        all_data_train = process_dict(data_train)
        all_data_dev = np.concatenate((all_data_train, all_data_dev))
        true_scores_dev = np.concatenate((true_scores_train, true_scores_dev))

    return all_data_dev, all_data_test, true_scores_dev, true_scores_test


def gradient_boosting():
    X_train, X_test, y_train, y_test = load_data(False, True)

    params = {
        'n_estimators': [50, 80, 100, 120],
        'subsample': [0.1, 0.2, 0.5, 1],
        'learning_rate': [0.01, 0.05, 0.1],
        'criterion': ['friedman_mse', 'mse'],
        'max_depth': [2, 3],
        'max_features': [3, 4, 5]
    }
    # {'criterion': 'mse', 'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 100, 'subsample': 0.1}
    regressor = GradientBoostingRegressor()
    #regressor.fit(X_train, y_train)
    gs = GridSearchCV(regressor, params, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    gs.fit(X_train, y_train)
    print('Best parameters: ')
    print(gs.best_params_)
    predictions = gs.predict(X_test)
    paired_predictions = np.split(predictions, 7)
    paired_targets = np.split(y_test, 7)
    base_predictions = [X_test[:, i] for i in range(0, 5)]
    base_paired_predictions = [np.split(p, 7) for p in base_predictions]
    pearson_scores = []
    print('---------Base--------')
    for i in range(len(base_paired_predictions)):
        scores = []
        for pred, target in zip(base_paired_predictions[i], paired_targets):
            scores.append(round(pearsonr(pred, target)[0], 2))
        print(scores)
    print('------------------')
    for pred, target in zip(paired_predictions, paired_targets):
         pearson_score = pearsonr(pred, target)[0]
         pearson_scores.append(round(pearson_score, 2))

    print(pearson_scores)


def stacking(regressor, use_train=False):
    X_train, X_test, y_train, y_test = load_data(False, use_train)
    models = {
        'linear': sklearn.linear_model.LinearRegression(),
        'ridge': sklearn.linear_model.Ridge(random_state=42),
        'svm': sklearn.svm.LinearSVR(random_state=42)
    }
    model = models[regressor]
    if regressor == 'svm':
        params = {
            'C': [0.1, 0.5, 1.0]
        }
        model = GridSearchCV(model, params, n_jobs=-1)
    if regressor == 'ridge':
        params = {
            'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]
        }
        model = GridSearchCV(model, params, n_jobs=-1, verbose=1)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    paired_predictions = np.split(predictions, 7)

    paired_targets = np.split(y_test, 7)

    base_predictions = [X_test[:, i] for i in range(0, 5)]
    base_paired_predictions = [np.split(p, 7) for p in base_predictions]
    pearson_scores = []
    if use_train:
        print('---------Base--------')
        for i in range(len(base_paired_predictions)):
            scores = []
            for pred, target in zip(base_paired_predictions[i], paired_targets):
                scores.append(round(pearsonr(pred, target)[0], 3))
            print(scores)
    print(regressor)
    print('------------------')
    for pred, target in zip(paired_predictions, paired_targets):
        pearson_score = pearsonr(pred, target)[0]
        pearson_scores.append(round(pearson_score, 3))

    print(pearson_scores)

if __name__ == "__main__":
    #check_predictions()
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    #config = yaml.load(open('configs\\predict_config.yaml'), Loader=yaml.FullLoader)
    #gradient_boosting()
    #print('Stacking')
    print("[en-de, en-zh, et-en, ne-en, ro-en, ru-en, si-en]")
    stacking('ridge')
    stacking('linear')
    stacking('svm')
    #stacking(True)
    #create_dataset(config)
    #load_data()