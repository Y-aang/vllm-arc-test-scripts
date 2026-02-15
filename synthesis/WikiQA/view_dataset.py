import dask.dataframe as dd

splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
# df = dd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits["test"])
df = dd.concat([
    dd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits[k])
    # for k in ["train", "validation", "test"]
    for k in ["test"]
])
df = dd.read_parquet("hf://datasets/Cohere/wikipedia-22-12-en-embeddings/data/train-*-of-*.parquet")

# Number of rows
num_rows = df.shape[0].compute()
print("Total rows:", num_rows)

# # Count unique document_title values
# unique_titles = df['document_title'].nunique().compute()
# print("Unique document_title count:", unique_titles)

# # print(df.head())
# row = df.head(10).iloc[1]   # Get the first row
# for col, value in row.items():
#     print(f"{col}: {value}\n")


