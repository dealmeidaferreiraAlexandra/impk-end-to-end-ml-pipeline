from __future__ import annotations

import numpy as np
import pandas as pd

TITLE_MAP = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Dr": "Dr",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Don": "Rare",
    "Lady": "Rare",
    "Countess": "Rare",
    "Jonkheer": "Rare",
    "Sir": "Rare",
    "Capt": "Rare",
    "Dona": "Rare",
    "Unknown": "Unknown",
}

def extract_title(name: object) -> str:
    if pd.isna(name):
        return "Unknown"
    name = str(name)
    if "," in name and "." in name:
        raw_title = name.split(",")[1].split(".")[0].strip()
        return TITLE_MAP.get(raw_title, "Rare")
    return "Unknown"

def extract_ticket_prefix(ticket: object) -> str:
    if pd.isna(ticket):
        return "None"
    ticket = str(ticket)
    letters = "".join(ch for ch in ticket if ch.isalpha())
    return letters[:4].upper() if letters else "None"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "Pclass": 3,
        "Name": "Unknown",
        "Sex": "unknown",
        "Age": np.nan,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "None",
        "Fare": np.nan,
        "Cabin": np.nan,
        "Embarked": "S",
    }

    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].fillna("U").astype(str).str[0].replace({"n": "U", "N": "U"})
    df["TicketPrefix"] = df["Ticket"].apply(extract_ticket_prefix)
    df["FarePerPerson"] = df["Fare"].fillna(0) / df["FamilySize"].replace(0, 1)

    df["Embarked"] = df["Embarked"].fillna("S").astype(str).str.upper()
    df["Sex"] = df["Sex"].astype(str).str.lower()
    df["Pclass"] = df["Pclass"].fillna(3).astype(int)

    return df