import pickle

# Read pickle file
data_file = "wikiqa_doc_query_dict.pkl"
# data_file = "/home/shenyang/data/doc_query_dict.pkl"
with open(data_file, "rb") as pfile:
    doc_query_dict = pickle.load(pfile)

# Count total entries (number of passages)
num_passages = len(doc_query_dict)

# Count total number of queries
num_queries = sum(len(queries) for queries in doc_query_dict.values())

# Print information
print(f"Total documents (passages): {num_passages}")
print(f"Total associated queries: {num_queries}")

# Preview sample data
print("\nSample data:")
for i, (passage, queries) in enumerate(doc_query_dict.items()):
    print(f"\nPassage {i+1}: {passage[:200]}...")  # Display first 200 characters
    print(f"Queries: {queries[:3]}")  # Show only first 3 queries
    if i == 2:  # Only preview 3 entries
        break

query_counts = [str(len(queries)) for queries in doc_query_dict.values()]
print(" ".join(query_counts))  # Space-separated, single line output