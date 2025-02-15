import os
import shutil
import pandas as pd
from tqdm import tqdm
from utils import summarize


def preprocess(counts=4):
    """
    This is a simple function to retain data used for the project.
    Check README.md for more details.
    """

    data = pd.read_csv("data/xray_data.csv")
    data.drop(["Unnamed: 0", "certainty", "diagnosis", "desc_pth"], axis=1, inplace=True)
    unique_ailements = ["Atelectasis", "Pneumonia", "Pleural Effusion", "Cardiomegaly", "Pneumothorax"]
    unique_counts = {ail: 0 for ail in unique_ailements}
    idx2keep = []

    # first find indices to keep
    for _, row in tqdm(data.iterrows(), desc="Finding indices to keep...", total=len(data)):
        labels = row["label"].split(",")
        labels = [label.strip() for label in labels]
        if len(labels) == 1:
            if labels[0] in unique_ailements:
                unique_counts[labels[0]] += 1
                if unique_counts[labels[0]] <= counts:
                    idx2keep.append(row["case"])

    data = data[data["case"].isin(idx2keep)]
    print(len(data))

    # now we iterate over again, and change the img_dir and summarize the reports
    for idx, row in tqdm(data.iterrows(), desc="Processing data...", total=len(data)):
        try:
            img_dir = "data/" + row["img_dir"]
            img = img_dir + "/" + os.listdir(img_dir)[0]
            # now we copy this image to "xrays"
            new_path = "data/xrays/" + img.split("/")[-1]
            shutil.copy(img, new_path)
            data.at[idx, "img_path"] = new_path
            report = row["desc"]
            ailment = row["label"]
            summary = summarize(report, ailment)
            data.at[idx, "report"] = summary
        except Exception as e:
            print(f"{e} at {idx}, {row['case']}.")
            continue

    data.drop(["img_dir", "desc"], axis=1, inplace=True)
    data.to_csv("data/xray_data.csv", index=False)

    return


if __name__ == "__main__":
    preprocess(counts=1000)
