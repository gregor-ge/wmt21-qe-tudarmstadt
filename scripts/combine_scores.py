from scipy.stats import pearsonr
import numpy as np
import os
import pandas as pd

base_path_xms = os.path.join('..', 'results', 'qe-da', 'xmoverscore')
base_path_reg = os.path.join("G:", "Meine Ablage", "Uni", "Seminar", "results", "qe-da", "ensemble", "train_xlmrl_all_base_r8_bs8_lr1-4_seed1qe_da_2021-06-09_05-49", "submissions")
files = {
    'en-de': '',
    'en-zh': '',
    'ru-en': '',
    'ro-en': ''
}
files = ['en-de', 'en-zh', 'ru-en', 'ro-en']
true_scores = []
for pair in files:
    lang1, lang2 = pair.split('-')
    da_scores = pd.read_csv(f"../data/data/direct-assessments/test/{lang1}-{lang2}/test20.{lang1}{lang2}.df.short.tsv",
                                    delimiter="\t", quoting=3)['z_mean']
    true_scores.append(da_scores)


def combine(xm_scores, reg_scores, alpha):
    weighted_xms = alpha * np.array(xm_scores)
    weighted_reg = (1 - alpha) * np.array(reg_scores)
    return weighted_reg + weighted_xms

def load_submission_file():
    submission_file = os.path.isfile(os.path.join('..', 'results', 'qe-da', 'predictions_xlmr_multi_all_.txt'))
    submission_file = 'predictions_xlmr_multi_all_.txt'
    submission_file = 'message.txt'
    temp = open(submission_file, 'r', encoding='utf-8').read().splitlines()[2:]
    lines = [t.split('\t') for t in temp]
    data = {}
    for pair in files:
        pair_data = []
        for line in lines:
            if line[0] == pair:
                pair_data.append(float(line[3]))
        data[pair] = pair_data
    return data

def load_scores():
    all_xm_scores = []
    all_reg_scores = []
    #for pair, submission_files in files.items():
    reg_data = load_submission_file()
    print()

    for i, pair in enumerate(files):
        xms_file = os.path.join(base_path_xms, pair + '.txt')
        l1, l2 = pair.split('-')
        #reg_file = os.path.join(base_path_reg, "predictions_" + l1 + "_" + l2 + ".txt")
        #with open(reg_file, 'r') as f:
        #    temp = f.read().splitlines()
        #    reg_scores = [float(s.split('\t')[-1]) for s in temp]
        reg_scores = reg_data[pair]
        with open(xms_file, 'r') as f:
            temp = f.read().splitlines()
            xm_scores = [float(s) for s in temp]
        all_xm_scores.append(xm_scores)
        all_reg_scores.append(reg_scores)
    return all_xm_scores, all_reg_scores


def main(alphas):
    xm_scores, reg_scores = load_scores()
    #regression_pearson = []
    #for i in range(len(reg_scores)):
    #    reg = reg_scores[i]
    #    gt = true_scores[i]
    #    pearson = pearsonr(reg, gt)
    lp = ['en-de', 'en-zh', 'ru-en', 'ro-en']

    reg_pearsons = [pearsonr(reg, gt.tolist())[0] for reg, gt in zip(reg_scores, true_scores)]
    for alpha in alphas:
        i = 0
        print('---------------- alpha: {} ----------------'.format(alpha))
        for xms, reg in zip(xm_scores, reg_scores):
            reg_corr = reg_pearsons[i]
            combined_scores = combine(xms, reg, alpha)
            combined_corr = pearsonr(combined_scores, true_scores[i])[0]

            if reg_corr < combined_corr or True:
                print("Regression corr: {}, Combined corr: {} ({})".format(reg_corr, combined_corr, lp[i]))
            i += 1

alphas = [0.1, 0.25, 0.5]

main(alphas)
