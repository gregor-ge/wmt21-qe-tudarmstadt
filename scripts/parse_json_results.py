import json
import os


ordering = {
    "si_en": 0,
    "ne_en": 1,
    "et_en": 2,
    "ro_en": 3,
    "en_de": 4,
    "en_zh": 5,
    "ru_en": 6
}

def parse(root, folders):
    devs = []
    tests = []
    for f in folders:
        folder = os.path.join(root, f)
        res = json.load(open(os.path.join(folder, "evaluation_qe_da.json")))

        # dev_res = [0]*len(ordering)
        # for r in res["dev"]:
        #     dev_res[ordering[r["pair"]]] = r["dev_pearson"]
        # devs.append(",".join([f"{p:.4f}" for p in dev_res]))

        test_res = [0]*len(ordering)
        for r in res["test"]:
            test_res[ordering[r["pair"]]] = r["test_pearson"]
        tests.append(",".join([f"{p:.4f}" for p in test_res]))

    print("\n".join(folders))
    # print("DEV")
    # print("\n".join(devs))
    print("TEST")
    print("\n".join(tests))


if __name__ == "__main__":

    root = r"E:\Programming\wmt2021-qe\results\sanity"
    folders = list(os.listdir(root))
    print(",".join(ordering.keys()))
    parse(root, folders)