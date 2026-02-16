import pickle
import json

def extract_fake_tokens(pkl_path: str, output_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)  # 这是一个 list[dict]

    # 提取所有 fake_tokens
    sentences = [item["fake_tokens"] for item in data if "fake_tokens" in item]

    # 保存成一行一条的 json list（如你给的例子）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=4)

    print(f"✅ Done! Saved {len(sentences)} entries to {output_path}")

if __name__ == "__main__":
    extract_fake_tokens("processed.pkl", "sentences.json")
