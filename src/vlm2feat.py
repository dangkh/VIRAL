import os
import requests
from math import nan
import json
import pandas as pd
import csv

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
# import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
import gzip
import ast
from config import TrainConfig 
from tqdm import tqdm
import yaml
import torch

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


linkmeta = ["https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz",
"https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
"https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz"]
link5cores = ["https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz",
"https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
"https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz"]

datasets = ["baby", "sport", "cloth"]



def getDescribe(vlmModel, processor, link = None, title = None, cfg = None, all_prompts = None):
    if cfg.template == 'title':
        myPrompt = all_prompts['vlm']['title'].format(title)
    else:
        myPrompt = all_prompts['vlm']['plain']
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": link,
            },
            {
                "type": "text",
                "text": (myPrompt)
            },
        ],
    }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = vlmModel.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


cfg = TrainConfig()
print(cfg)

# CRAWLING DATA
myIDX = datasets.index(cfg.data)
crawlData = cfg.data
url = linkmeta[myIDX]

filePath = f'./data/{crawlData}/{crawlData}.json.gz'
if os.path.exists(filePath):
    print("✅ File exist.")
else:
  response = requests.get(url, stream=True)
  if response.status_code == 200:
      with open(filePath, 'wb') as f:
          f.write(response.raw.read())
      print("Download complete.")
  else:
      print(f"Failed to download. Status code: {response.status_code}")


url = link5cores[myIDX]
filePath = f'./data/{crawlData}/review_{crawlData}.json.gz'
if os.path.exists(filePath):
    print("✅ File exist.")
else:
  response = requests.get(url, stream=True)
  if response.status_code == 200:
      with open(filePath, 'wb') as f:
          f.write(response.raw.read())
      print("Download complete.")
  else:
      print(f"Failed to download. Status code: {response.status_code}")


data = []
with gzip.open(f'./data/{crawlData}/{crawlData}.json.gz', 'rt') as f:
    for line in f:
        data.append(ast.literal_eval(line))

metaDF = pd.DataFrame(data)
metaDF_filtered = metaDF[["asin", "title", "brand", "description", "imUrl", "categories"]].copy()

unique_Meta_asin = metaDF_filtered['asin'].unique()
print(f"Number of unique ASINs: {len(unique_Meta_asin)}")
data = []
with gzip.open(f"./data/{crawlData}//review_{crawlData}.json.gz", "r") as f:
  for line in f:
    data.append(json.loads(line))

review5DF = pd.DataFrame(data)
print(review5DF.columns)

unique_asin = review5DF['asin'].unique()
print(f"Number of unique ASINs: {len(unique_asin)}")

location = f"./data/{cfg.data}/{cfg.data}_product_images"

counter = 0
counter_error = 0


os.makedirs(f"{location}", exist_ok=True)


for cnt, row in tqdm(metaDF_filtered.iterrows(), total =  len(metaDF_filtered)):
  image_urls = row['imUrl']
  asin = row['asin']
  if asin not in unique_asin:
    continue
  if os.path.exists(f"{location}/{asin}.jpg"):
    continue
  try:
    img_data = requests.get(image_urls).content
    with open(f"{location}/{asin}.jpg", 'wb') as f:
      f.write(img_data)
  except Exception as e:
    counter += 1


print(f"missing: {counter}")

num_files = len(os.listdir(f"{location}"))
print(f"Number of files in product_images: {num_files}")

image_directory = location
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
len(image_files)


# Create a dictionary to map ASINs to image paths
asin_image_paths = {}
tmp = []
for image_path in image_files:
    # Assuming the filename is the ASIN followed by .jpg
    asin = os.path.splitext(os.path.basename(image_path))[0]
    tmp.append(asin)
    asin_image_paths[asin] = image_path

# Create a dictionary to store the descriptions for unique ASINs
asin_descriptions = {}
len(asin_image_paths)



# DESCRIPTION FROM VLM
with open("src/prompts.yaml", "r") as f:
    all_prompts = yaml.safe_load(f)


if cfg.vlmModel == 'qwen':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
elif cfg.vlmModel == 'blip':
    pass
elif cfg.vlmModel == 'gema':
    pass
else:
    model_path = "lmms-lab/LLaVA-One-Vision-1.5-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)




counter = 0

# Iterate through the unique ASINs and get descriptions for corresponding images
for asin in tqdm(unique_asin):
    if asin in asin_descriptions:
        continue
    if asin in asin_image_paths:
        image_path = asin_image_paths[asin]
        try:
            product_row = metaDF_filtered[metaDF_filtered['asin'] == asin]
            title = product_row['title'].iloc[0]
            description = product_row['description'].iloc[0]
            description = getDescribe(model, processor, image_path, title, cfg, all_prompts)
            
            asin_descriptions[asin] = description[0] if description else nan
            counter += 1
        except Exception as e:
            print(f"Error processing ASIN {asin}: {e}")
            asin_descriptions[asin] = nan
    else:
        asin_descriptions[asin] = nan # ASIN not found in downloaded images
    if len(asin_descriptions) % 100 == 0:
        # Convert the results to a list of tuples for writing to CSV
        amazon_res = [(asin, desc) for asin, desc in asin_descriptions.items()]

        # Write the results to a CSV file
        amazon_output_filename = f'./data/{cfg.data}/amazon_{cfg.data}_model_{cfg.vlmModel}_type_{cfg.template}_descriptions.csv'
        with open(amazon_output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['asin', 'description']) # Write header
            writer.writerows(amazon_res)

        print(f"Amazon descriptions saved to {amazon_output_filename}")        























