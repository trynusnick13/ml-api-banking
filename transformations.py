import pickle
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def transform_user_details_to_scalars(user: Dict[str, Any]) -> int:
    df = pd.read_csv('train.csv', delimiter=";")
    df = df[df.columns[:-1]]
    df = df.append(user, ignore_index=True)

    df['job'] = df['job'].str.replace('.', '')
    df.loc[(df.job == "unknown") & (df.education == "secondary"), "job"] = "services"
    df.loc[(df.job == "unknown") & (df.education == "primary"), "job"] = "housemaid"
    df.loc[(df.job == "unknown") & (df.education == "tertiary"), "job"] = "management"
    df.loc[(df.job == "unknown"), "job"] = "blue-collar"
    df.loc[(df.education == "unknown") & (df.job == "admin"), "education"] = "secondary"
    df.loc[(df.education == "unknown") & (df.job == "management"), "education"] = "secondary"
    df.loc[(df.education == "unknown") & (df.job == "services"), "education"] = "tertiary"
    df.loc[(df.education == "unknown") & (df.job == "technician"), "education"] = "secondary"
    df.loc[(df.education == "unknown") & (df.job == "retired"), "education"] = "secondary"
    df.loc[(df.education == "unknown") & (df.job == "blue-collar"), "education"] = "secondary"
    df.loc[(df.education == "unknown") & (df.job == "housemaid"), "education"] = "primary"
    df.loc[(df.education == "unknown") & (df.job == "self-employed"), "education"] = "tertiary"
    df.loc[(df.education == "unknown") & (df.job == "student"), "education"] = "secondary"
    df.loc[(df.education == "unknown") & (df.job == "entrepreneur"), "education"] = "tertiary"
    df.loc[(df.education == "unknown") & (df.job == "unemployed"), "education"] = "secondary"
    df.loc[(df.education == "unknown"), "education"] = "secondary"
    df["contact"].replace(["unknown"], df["contact"].mode(), inplace=True)
    df.balance.sort_values()
    df.loc[(df.balance > 81204), "balance"] = 81204
    df["balance"] = df["balance"] / 81204
    df.drop(columns=["day"], inplace=True)
    ohe = OneHotEncoder(sparse=False)
    df = pd.concat(
        (
            df, pd.DataFrame(ohe.fit_transform(df["job"].to_frame()), columns="job_" + np.sort(df["job"].unique()))
        ),
        axis=1
    )
    df.drop(columns=["job"], inplace=True)

    df = pd.concat((df, pd.DataFrame(ohe.fit_transform(df["marital"].to_frame()),
                                     columns="marital_" + np.sort(df["marital"].unique()))), axis=1)
    df.drop(columns=["marital"], inplace=True)

    df = pd.concat((df, pd.DataFrame(ohe.fit_transform(df["poutcome"].to_frame()),
                                     columns="poutcome_" + np.sort(df["poutcome"].unique()))), axis=1)
    df.drop(columns=["poutcome"], inplace=True)

    lst = ['default', 'housing', 'loan']
    for i in lst:
        df[i] = df[i].map({'yes': 1, 'no': 0})

    df['education'] = df['education'].map({'tertiary': 2, 'secondary': 1, 'primary': 0})

    df['contact'] = df['contact'].map({'cellular': 0, 'telephone': 1})
    df.drop(columns=['month', 'duration'], inplace=True)
    df.loc[(df.pdays == -1), "pdays"] = 999
    new_cols = [col for col in df.columns]
    df = df[new_cols]
    x = df

    filename = "finalized_model.sav"
    path = Path(filename)

    if not path.is_file():
        shutil.unpack_archive(f"{filename}.zip")
    with open(filename, "rb") as model_file:
        model = pickle.load(model_file)
        result = model.predict(x[-1:])
    return result[0]
