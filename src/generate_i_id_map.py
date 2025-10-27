import pandas as pd
import json
from tqdm import tqdm

# Load data
data = [json.loads(line) for line in open("sports_5.json", "r")]
df = pd.DataFrame(data)

# Encode item IDs
unique_items = df['asin'].unique()
i2id = {asin: i for i, asin in enumerate(unique_items)}

# Save mapping
pd.DataFrame(list(i2id.items()), columns=['asin', 'item_id']).to_csv('i_id_mapping.csv', index=False)
