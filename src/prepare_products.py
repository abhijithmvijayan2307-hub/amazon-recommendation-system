import pandas as pd
import os
import ast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "..", "data", "meta_Electronics.json")
output_path = os.path.join(BASE_DIR, "..", "data", "products.csv")

print("Reading metadata from:", input_path)

data = []

with open(input_path, 'r', encoding='utf8') as f:
    for i, line in enumerate(f):
        try:
            obj = ast.literal_eval(line)

            if isinstance(obj, dict) and 'asin' in obj and 'title' in obj:
                title = obj['title'].strip()
                if title != "":
                    data.append((obj['asin'], title))

        except:
            continue

        if i % 200000 == 0:
            print(i, "rows processed")

df = pd.DataFrame(data, columns=['asin', 'title'])
df.drop_duplicates(inplace=True)

df.to_csv(output_path, index=False)

print("\nSaved products.csv")
print("Total products:", len(df))
