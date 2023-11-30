import pandas as pd
from datasets import load_dataset
import os


def process_data(dataset, prefix=None, suffix=None, frac=0.7, seed=42, save_path="data"):
    train = dataset.sample(frac=frac, random_state=seed).copy()
    valid = dataset[~dataset.index.isin(train.index)].copy()

    print(train, valid)


    if prefix is not None:
        train = [prefix + str(t) for t in train]
        valid = [prefix + str(v) for v in valid]

    if suffix is not None:
        train = [str(t) + suffix for t in train]
        valid = [str(v) + suffix for v in valid]

    with open("./data/train.txt", "w") as file:
        file.write("\n".join(map(str, train)))

    with open(os.path.join(save_path, "valid.txt"), "w") as file:
        file.write("\n".join(map(str, valid)))

def get_df():

    dataset = load_dataset("blinoff/medical_qa_ru_data")
    df = pd.DataFrame(dataset['train'])


    df = df[df["categ"] == "Травматология и ортопедия"]
    #df = df.sample(3000)

    df["concated_data"] = "<s>Категория: " + df["categ"] + " --> " + "Вопрос: " + df['desc'] + " ==> " + "Ответ: " + df["ans"] + "</s>"
    return df['concated_data']

process_data(get_df())
