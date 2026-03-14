import pandas as pd
import json
from tqdm import tqdm
import gzip

# Load data
# data = [json.loads(line) for line in open("./data/sports/sports_5.json", "r")]
# df = pd.DataFrame(data)

# # Encode item IDs
# unique_items = df['asin'].unique()
# i2id = {asin: i for i, asin in enumerate(unique_items)}

# # Save mapping
# pd.DataFrame(list(i2id.items()), columns=['asin', 'item_id']).to_csv('i_id_mapping.csv', index=False)


data = []
with gzip.open(f"./data/cloth/review_cloth.json.gz", "r") as f:
  for line in f:
    data.append(json.loads(line))

review5DF = pd.DataFrame(data)
# Encode item IDs
unique_items = review5DF['asin'].unique()
i2id = {asin: i for i, asin in enumerate(unique_items)}
print(f"Total unique items: {len(unique_items)}")

# Save mapping
pd.DataFrame(list(i2id.items()), columns=['asin', 'item_id']).to_csv('./data/cloth/i_id_mapping.csv', index=False)
