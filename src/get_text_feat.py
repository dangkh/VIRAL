import argparse
import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import ast

def printHead(df, note = "SAMPLE"):
    print(f"{note}: ")
    print(df.head()) 

def arg_parse():
    parser = argparse.ArgumentParser(description="Get text features a dataset.")
    parser.add_argument("--dataset", type=str, required=True, 
                        default="baby", help="Name of the dataset.")
    parser.add_argument("--text_column", type=str, 
                        default="title", help="Name of the column containing text data.")
    parser.add_argument("--vlm", type=str, default="qwen", help="Name of vlm model.")
    parser.add_argument("--type_prompt", type=str, 
                        default="title", 
                        help="Name of the prompt generateing data: title, plain, something")
    parser.add_argument("--add_meta", type=bool, default=True, help="add meta or not")
    parser.add_argument("--txt_embedding_model", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Name of the text embedding model to use.")
    args = parser.parse_args()
    return args

def get_encoding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

def get_dataset(args) -> dict: 
    dataset_folder = {
        "inter_file": f"data/{args.dataset}/{args.dataset}.inter",
        "mapping_file": f"data/{args.dataset}/i_id_mapping.csv",
        "description_file": 
            f"data/{args.dataset}/amazon_{args.dataset}_model_{args.vlm}_type_{type_prompt}_descriptions.csv.json",
        "meta_data": f"data/{args.dataset}/meta_{args.dataset}.json",
        "five_core_data": f"data/{args.dataset}/{args.dataset}_5.json",
    }
    return dataset_folder

def get_details_dataset_df(args, mapping_file, five_core_data, description_file) -> pd.DataFrame:
    
    with open(os.path.join(description_file), "r", encoding="utf-8") as f:
        description_data = json.load(f)

    description_df = pd.DataFrame(description_data)
    i_id_mapping = pd.read_csv(os.path.join(mapping_file), sep="\t")

    ### map item with description sample on asin 
    item_id_with_description_df = i_id_mapping.merge(description_df, on="asin", how="left")
    for id, row in item_id_with_description_df.iterrows():
        if pd.isna(row[args.type_prompt]):
            item_id_with_description_df.at[id, args.type_prompt] = ""
    
    printHead(item_id_with_description_df)
    return item_id_with_description_df

def concat_with_meta_data(args, df, meta_data_file, text_column):
    """
    Concatenate the DataFrame with metadata from the specified file.
    """
    meta_data = []
    with open(os.path.join(meta_data_file), "r") as f:
        for line in f:
            # Safely evaluate the string literal before loading as JSON
            meta_data.append(json.loads(json.dumps(ast.literal_eval(line.strip()))))

    meta_df = pd.DataFrame(meta_data)
    meta_df_5_core = df.merge(meta_df, on="asin", how="left")
    meta_df_5_core.rename(columns={"title_x": "title"}, inplace=True)

    for id, row in enumerate(meta_df_5_core[args.type_prompt]):
      if pd.isna(row):
        meta_df_5_core.loc[id, args.type_prompt] = meta_df_5_core["description"][id]

    ### Concat
    for id, row in meta_df_5_core.iterrows():
        for col in ["title", "brand"]:
            if pd.isna(row[col]):
                meta_df_5_core.at[id, col] = "" 

    if args.add_meta:                
        meta_df_5_core["visual_enriched"] = meta_df_5_core[text_column]  + ". " + \
                                    meta_df_5_core[args.type_prompt]

    printHead(meta_df_5_core, "SAMPLE all info")
    return meta_df_5_core

def get_text_features(args, model, df):
    """
    Get text features for the specified text column in the DataFrame.
    """
    txt_embeddings = model.encode(df["visual_enriched"], show_progress_bar=True, normalize_embeddings=True)
    print(f"SHAPE of matrix: {txt_embeddings.shape}")
    np.save(os.path.join(f"./data/{args.dataset}/text_feat.npy"), txt_embeddings)

def main():
    args = arg_parse()

    if not os.path.exists("data"):
        raise FileNotFoundError("Could not find the data directory. Please ensure it exists.")

    if not os.path.exists(os.path.join("data", args.dataset)):
        raise FileNotFoundError(f"Could not find the dataset directory for {args.dataset}. Please ensure it exists.")
    
    model = get_encoding_model(args.txt_embedding_model)
    dataset_folder = get_dataset(args)
    detail_dataset_df = get_details_dataset_df(args, dataset_folder["mapping_file"], 
                                               dataset_folder["five_core_data"], 
                                               dataset_folder["description_file"])
    print(f" Number of items: {len(detail_dataset_df)}")
    detail_dataset_df = concat_with_meta_data(args, 
                                                    detail_dataset_df,
                                                    dataset_folder["meta_data"], 
                                                    args.text_column)
    # else:
    #     detail_dataset_df['visual_enriched'] = detail_dataset_df[args.type_prompt]
    

    get_text_features(args, 
                      model=model,
                      df=detail_dataset_df)

    detail_dataset_df.to_csv(f"./data/{args.dataset}/allMeta_data.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()

