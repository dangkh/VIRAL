
# VLIF

## Data Preprocess


Example for dataset "baby":
```
data/baby/
├── amazon_description_baby_sample.json
├── baby_5.json
├── baby.inter
├── i_id_mapping.csv
├── meta_baby.json
└── text_feat.npy
```

## Usage

### 1. Install required libraries

```sh
pip install -r requirements.txt
```

### 2. Generate text features

Example with the "baby" dataset, using the "title" column as text data:
```sh
python get_text_feat.py --dataset=baby --text_column=title
```

Arguments:
- `--dataset`: Name of the dataset (e.g., baby)
- `--text_column`: Name of the column containing text data (e.g., title)
- `--txt_embedding_model` (Not required): Name of the embedding model for text data (e.g., sentence-transformers/all-MiniLM-L6-v2)

The embedding file will be saved at: `data/<dataset_name>/<text_column>_txt_feat.npy`
