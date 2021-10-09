import streamlit as st
import numpy as np
import datetime
import pandas as pd
import lightgbm as lgb
#from sklearn.metrics import roc_auc_score
from urllib.request import urlopen
import time
#from tqdm.notebook import tqdm
from stqdm import stqdm
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup
import re
#import optuna.integration.lightgbm as lgb_opt
import matplotlib.pyplot as plt
import pickle


class DataProcessor:
    def __init__(self):
        self.data = pd.DataFrame()
        self.data_p = pd.DataFrame()  # preprocess
        self.data_h = pd.DataFrame()  # merge horse_results
        self.data_pe = pd.DataFrame()  # merge peds
        self.data_c = pd.DataFrame()  # process categorical

    def merge_horse_results(self, hr, n_samples_list=[5, 9, "all"]):
        self.data_h = self.data_p.copy()
        for n in n_samples_list:
            self.data_h = hr.merge_all(self.data_h, n_samples=n)

    def merge_peds(self, peds):
        self.data_pe = self.data_h.merge(
            peds, left_on="horse_id", right_index=True, how="left"
        )
        self.no_peds = self.data_pe[self.data_pe["peds_0"].isnull()][
            "horse_id"
        ].unique()  # 血統データをスクレイピングしていない馬
        if len(self.no_peds) > 0:
            print("新しい馬の血統データを後でスクレイピング")

    def process_categorical(self, le_horse, le_jockey, results_m):
        df = self.data_pe.copy()
        #新しいカテゴリが入ってきたときのラベルエンコーディング
        # classesに存在しないhorse_idをclassesに追加する
        mask_horse = df["horse_id"].isin(le_horse.classes_)
        new_horse_id = df["horse_id"].mask(
            mask_horse).dropna().unique()  # type:numpy array
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        df["horse_id"] = le_horse.transform(df["horse_id"])

        mask_jockey = df["jockey_id"].isin(le_jockey.classes_)
        new_jockey_id = df["jockey_id"].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate(
            [le_jockey.classes_, new_jockey_id])
        df["jockey_id"] = le_jockey.transform(df["jockey_id"])

        df["horse_id"] = df["horse_id"].astype("category")
        df["jockey_id"] = df["jockey_id"].astype("category")

        #ダミー変数化　例えばそのレースが曇りだと、晴れの列ができないー＞pandasのcategoricalを使ってカテゴリ型にして列数をあわせる
        weathers = results_m["weather"].unique()
        race_types = results_m["race_type"].unique()
        ground_states = results_m["ground_state"].unique()
        sexes = results_m["性"].unique()
        # 第１引数に変換したいデータを、第２引数（categories）に存在するカテゴリを指定する
        df["weather"] = pd.Categorical(df["weather"], weathers)
        df["race_type"] = pd.Categorical(df["race_type"], race_types)
        df["ground_state"] = pd.Categorical(df["ground_state"], ground_states)
        df["性"] = pd.Categorical(df["性"], sexes)
        #columnsでダミー変数化する列を指定
        # categoricalでカテゴリを指定した上でダミー変数化するとそのレースの天候以外の天候（性なども同じ）もダミー変数化される
        df = pd.get_dummies(
            df, columns=["weather", "race_type", "ground_state", "性"])

        self.data_c = df


class ShutubaTable(DataProcessor):
    def __init__(self, shutuba_tables):
        super(ShutubaTable, self).__init__()
        self.data = shutuba_tables

    @classmethod
    def scrape(cls, race_id_list, date):
        data = pd.DataFrame()
        for race_id in race_id_list:
            url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + race_id
            df = pd.read_html(url)[0]
            # マルチインデックスを解除　reset_indexで指定した階層のインデックスを解除して列にもってこれてドロップもできる
            df = df.T.reset_index(level=0, drop=True).T
            html = requests.get(url)
            html.encoding = "EUC-JP"
            soup = BeautifulSoup(html.text, "html.parser")

            texts = soup.find("div", attrs={"class": "RaceData01"}).text
            texts = re.findall(r"\w+", texts)
            for text in texts:
                if "m" in text:
                    df["corse_len"] = [
                        int(re.findall(r"\d+", text)[0])] * len(df)
                if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                    df["weather"] = [text] * len(df)
                if text in ["良", "稍重", "重", "稍"]:
                    df["ground_state"] = [text] * len(df)
                if text in "不":
                    df["ground_state"] = ["不良"] * len(df)
                if "芝" in text:
                    df["race_type"] = ["芝"] * len(df)
                if "ダ" in text:
                    df["race_type"] = ["ダート"] * len(df)
                if "障" in text:
                    df["race_type"] = ["障害"] * len(df)
            df["date"] = [date] * len(df)

            # horse_id
            horse_id_list = []
            horse_td_list = soup.find_all("td", attrs={"class": "HorseInfo"})
            for td in horse_td_list:
                horse_id = re.findall(r"\d+", td.find("a")["href"])[0]
                horse_id_list.append(horse_id)
            # jockey_id
            jockey_id_list = []
            jockey_td_list = soup.find_all("td", attrs={"class": "Jockey"})
            for td in jockey_td_list:
                jockey_id = re.findall(r"\d+", td.find("a")["href"])[0]
                jockey_id_list.append(jockey_id)

            df["horse_id"] = horse_id_list
            df["jockey_id"] = jockey_id_list

            df.index = [race_id] * len(df)
            data = data.append(df)
            time.sleep(1)
        return cls(data)

    def preprocessing(self, flag=False):  # 馬体重を除く
        df = self.data.copy()
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        if flag:
            df.drop(["馬体重(増減)"], axis=1, inplace=True)
        else:
            df = df[df["馬体重(増減)"] != "--"]
            df = df[~df["馬体重(増減)"].isnull()]
            # 馬体重を体重と体重変化に分ける
            df["体重"] = df["馬体重(増減)"].str.split("(", expand=True)[0].astype(int)
            df["体重変化"] = (
                df["馬体重(増減)"].str.split(
                    "(", expand=True)[1].str[:-1].astype(int)
            )

        df["date"] = pd.to_datetime(df["date"])
        df["枠"] = df["枠"].astype(int)
        df["馬番"] = df["馬番"].astype(int)
        df["斤量"] = df["斤量"].astype(int)
        df_list = [
            "枠",
            "馬番",
            "斤量",
            "corse_len",
            "weather",
            "race_type",
            "ground_state",
            "date",
            "horse_id",
            "jockey_id",
            "性",
            "年齢",
        ]
        if not flag:
            df_list.append("体重")
            df_list.append("体重変化")
        # 不要な列を削除
        df = df[df_list]
        self.data_p = df.rename(columns={"枠": "枠番"})


class HorseResults:  # 着順と賞金の平均を扱う
    def __init__(self, horse_results):
        self.horse_results = horse_results[["日付", "着順", "賞金", "着差", "通過"]]
        self.preprocessing()
        # self.horse_results.rename(columns={'着順':'着順（平均）','賞金':'賞金（平均）'},inplace=True)

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.concat([pd.read_pickle(path) for path in path_list])
        return cls(df)

    def preprocessing(self):
        df = self.horse_results.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
        df.dropna(subset=["着順"], inplace=True)
        df["着順"] = df["着順"].astype(int)

        df["date"] = pd.to_datetime(df["日付"])
        df.drop(["日付"], axis=1, inplace=True)
        df["賞金"].fillna(0, inplace=True)

        df["着差"] = df["着差"].map(lambda x: 0 if x < 0 else x)

        df["通過"] = df["通過"].str.split("-")
        df["first_corner"] = df["通過"].map(
            lambda x: int(x[0]) if type(x) == list else x)
        df["final_corner"] = df["通過"].map(
            lambda x: int(x[-1]) if type(x) == list else x)

        df["final_to_rank"] = df["final_corner"] - df["着順"]
        df["first_to_rank"] = df["first_corner"] - df["着順"]
        df["first_to_final"] = df["first_corner"] - df["final_corner"]

        self.horse_results = df

    def average(self, horse_id_list, date, n_samples="all"):
        target_df = self.horse_results.query(
            "index in @horse_id_list"
        )  # horse_resultsにない新しい馬が出てきたときにエラーが起きないように

        if n_samples == "all":
            filtered_df = target_df[target_df["date"] < date]
        elif n_samples > 0:
            filtered_df = (
                target_df[target_df["date"] < date]
                .sort_values("date", ascending=False)
                .groupby(level=0)
                .head(n_samples)
            )
        else:
            raise Exception("正の値のみ")

        ave = filtered_df.groupby(level=0)[["着順", "賞金", "着差", "first_corner",
                                            "first_to_rank", "first_to_final",
                                            "final_to_rank", "final_corner"]].mean()
        return ave.rename(
            columns={
                "着順": "着順平均{}R分".format(n_samples),
                "賞金": "賞金平均{}R分".format(n_samples),
                "着差": "着差平均{}R分".format(n_samples),
                "first_corner": "first_corner{}R分".format(n_samples),
                "first_to_rank": "first_to_rank{}R分".format(n_samples),
                "first_to_final": "first_to_final{}R分".format(n_samples),
                "final_to_rank": "final_to_rank{}R分".format(n_samples),
                "final_corner": "final_corner{}R分".format(n_samples)
            }
        )

    def merge(self, results, date, n_samples="all"):
        df = results[results["date"] == date]
        horse_id_list = df["horse_id"]
        merged_df = df.merge(
            self.average(horse_id_list, date, n_samples),
            left_on="horse_id",
            right_index=True,
            how="left",
        )
        return merged_df

    def merge_all(self, results, n_samples="all"):
        date_list = results["date"].unique()
        merged_df = pd.concat(
            [self.merge(results, date, n_samples) for date in date_list]
        )
        return merged_df


def predict_proba(model, X):
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    # def standard_scaler(x): 
    #     return (x - x.mean()) / x.std()
    # proba = proba.groupby(level=0).transform(standard_scaler)
    # proba = (proba - proba.min()) / (proba.max() - proba.min())
    return proba

def main():
    st.title("中央競馬予想App")
    st.markdown("#### 競争馬が3着以内に入る確率を学習済みモデルで求めます")
    #r.data_c.head()
    target_rid = st.sidebar.text_input(
        label="予想したいレースのIDを入力(URLがhttps://race.netkeiba.com/race/result.html?race_id=202106040711の場合は202106040711)")
    target_date = st.sidebar.text_input(label="予想したいレースの開催日を入力(ex:2021/01/01)")
    flag = st.sidebar.checkbox(label = "馬体重を予想に使わない",value = False)
    #学習済みモデルの読み込み
    with open("lgb_model.pickle", "rb") as f:
        lgb_clf = pickle.load(f)
    if flag:
        with open("lgb_model_noweigh.pickle", "rb") as f:
            lgb_clf = pickle.load(f)
    
    flag_st = st.sidebar.checkbox(label = "予想を開始",value = False)

    if not flag_st:
        st.markdown('''
            ## 使い方
            - https://race.netkeiba.com/top/race_list.html　へアクセスして予想したいレースを選択
            - すると、URLに"race_id="という文字列が含まれているので、それに続く12桁の数字をコピペして左に入力
            - レースの開催日をoooo/oo/ooというフォーマットで左に入力する
            - 馬体重はレース開始の約30分前まで発表されないので、その場合はチェックして取り除く
            - 予想を開始する

        ''')

    if flag_st and len(target_rid) != 0 and len(target_date) != 0:
        stb = ShutubaTable.scrape([target_rid],target_date)
        stb.preprocessing(flag)
        with open("hrform.pickle", "rb") as f:
            hr = pickle.load(f)
        stb.merge_horse_results(hr)
        with open("allpeds.pickle","rb") as f:
            peds_df=pickle.load(f)
        stb.merge_peds(peds_df)
        with open("le_h.pickle", "rb") as f:
            le_h = pickle.load(f)
        with open("le_j.pickle", "rb") as f:
            le_j = pickle.load(f)
        with open("le_d.pickle", "rb") as f:
            le_d = pickle.load(f)
        stb.process_categorical(le_h,le_j,le_d)
        pred = predict_proba(lgb_clf,stb.data_c.drop(["date"], axis=1))
        pred_table = stb.data_c[["馬番"]].copy()
        pred_table["pred"] = pred
        pred_table=pred_table.sort_values("pred", ascending=False).loc[target_rid]
        pred_table.reset_index(inplace=True)
        pt = pred_table.drop("index",axis=1).set_index("馬番",drop=True)
        pt["pred"] = pt["pred"].map(lambda x: str(x * 100) + "%")
        st.table(pt)

if __name__ == "__main__":
    main()
