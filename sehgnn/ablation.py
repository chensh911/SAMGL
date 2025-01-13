# coding=utf-8

import numpy as np
import os
import re
import pandas as pd
# from matplotlib import pyplot as plt
import pandas as pd
import json

output_dir = "outputs"


def parse_file(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        first_line = f.readline()
        meta = json.loads(first_line)

        # print(fpath)
        # print("meta_seed = ", meta["seed"])

        line = None
        for line in f:
            pass

        if line is None:
            return None
        
        result = json.loads(line)

        if "early_stop_epoch" not in result:
            return None

        result = {
            **meta,
            **result
        }
    return result


results = []

for root, dirs, fnames in os.walk("./outputs"):
    for fname in fnames:
        if fname.endswith(".json"):
            fpath = os.path.join(root, fname)
            result = parse_file(fpath)
            
            if result is not None:
                results.append(result)
                # print(result)
                # print(result["seed"])
            else:
                print("skip result: {}".format(fpath))

# asdfasdf



df = pd.DataFrame.from_dict(results)



df = df.drop("date", axis=1)




columns = df.columns.values.tolist()
group_columns = ["dataset", "method", "use_nrl", 
                #  "use_gnn_nrl", "use_mix", 
                 "use_label", "even_odd", 
                #  "lp_k", "lp_a", 
                 "use_all_feat", "train_strategy", 
                #  "stage_strategy", 
                 "use_input", "input_drop_rate", "drop_rate", "hidden_size", "squash_k", "num_epochs", "max_patience", "embedding_size", "rps"]

invalid_group_columns = [column for column in group_columns if df[column].isna().all()]


for column in invalid_group_columns:
    df = df.drop(column, axis=1)
group_columns = [column for column in group_columns if column not in invalid_group_columns]


count_df = df.groupby(group_columns)["method"].count().rename("count").reset_index()
# df = df.groupby(group_columns).mean().reset_index()




df = df.groupby(group_columns).agg(["mean", "std"]).reset_index()

df.columns = [f'{col}_{stat}' if stat not in ['', 'mean'] else col for col, stat in df.columns]


df = pd.merge(df, count_df, how="left", left_on=group_columns, right_on=group_columns)

df = df.round(4)
df = df.sort_values(by='es_eval_nMSE')


shown_columns = [
        # "dataset", 
        #      "method", 
            #  "use_nrl", "use_gnn_nrl", "use_mix", 
            #  "train_strategy", 
            #  "stage_strategy", 
            #  "use_label", "lp_k", "lp_a", "use_all_feat", 
            #  "use_input", 
             
             "even_odd",
             "rps",
             "input_drop_rate", "drop_rate", "hidden_size", "squash_k", "num_epochs", "max_patience", "embedding_size", "count",
            # "imp",
    "pre_compute_time", "train_time", "all_time",
    "early_stop_epoch",

    # "es_val_accuracy",
    # "es_eval_accuracy", 
    # "eval_prop_accuracy", 

    # "es_val_macro_f1", "es_val_micro_f1",
    # "es_eval_macro_f1", "es_eval_micro_f1",


    "es_eval_MAE",
    "es_eval_nMSE",
    "es_eval_SRC",

    # "es_val_ndcg", "es_val_mrr",
    # "es_eval_ndcg", "es_eval_mrr",
    
    # "eval_prop_macro_f1", "eval_prop_micro_f1",
    # "eval_macro_f1_score", "eval_micro_f1_score"
]

shown_columns = shown_columns + ["{}_std".format(col) for col in shown_columns if col not in group_columns and col not in ["count"]]

# shown_columns = [column for column in shown_columns if column in df.columns]



with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):
    df = df[shown_columns]
    
    # df.to_csv("results_link_pred_ours.csv")
    # exit()

    for column in ["num_epochs", "max_patience", "hidden_size", "squash_k", "embedding_size", "lp_k"]:
        if column in df.columns:
            df[column] = df[column].astype("int32")

    
    rename_dict = {
        "use_input": "input", 
        "use_nrl": "nrl", 
        "use_mix": "mix",
        "use_label": "label", 
        "even_odd": "eo",
        "lp_k": "lpk",
        "lp_a": "lpa", 
        "use_all_feat": "afeat",
        "train_strategy": "t_s",
        "stage_strategy": "s_s", 
        "use_gnn_nrl":"gnrl",
        "input_drop_rate": "idr", 
        "drop_rate": "dr", 
        "hidden_size": "hs", 
        "squash_k": "k",
        "num_epochs": "ep", 
        "max_patience": "mpa",
        "count": "c", 
        "epochs": "ep",
        "early_stop_epoch": "ese",
        "embedding_size": "es",
        "pre_compute_time": "pt",
        "train_time": "tt",
        "all_time": "at",
        "dataset": "ds",
        # "eval_macro_f1_score": "eval_macro_f1",
        # "eval_micro_f1_score": "eval_micro_f1"


        "es_val_ndcg": "v_ndcg",
        "es_eval_ndcg": "t_ndcg",

        "es_val_mrr": "v_mrr",
        "es_eval_mrr": "t_mrr",

        "es_val_accuracy": "v_acc",
        "es_eval_accuracy": "t_acc",

        "es_val_macro_f1": "v_macro",
        "es_val_micro_f1": "v_micro",
        "es_val_prop_macro_f1": "v_p_macro", 
        "es_val_prop_micro_f1": "v_p_micro",

        "es_eval_macro_f1": "t_macro",
        "es_eval_micro_f1": "t_micro",
        "es_eval_prop_macro_f1": "t_p_macro", 
        "es_eval_prop_micro_f1": "t_p_micro",

    
    }

    std_rename_dict = {
        "{}_std".format(k):"{}_std".format(v) for k, v in rename_dict.items()
    }


    rename_dict = {**rename_dict, **std_rename_dict}

    df = df.rename(rename_dict, axis=1)
    print(df)
    # print(df.to_json())

   