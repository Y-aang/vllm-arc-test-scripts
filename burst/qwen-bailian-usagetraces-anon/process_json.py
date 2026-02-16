import json
import pickle
import random
from tqdm import tqdm

def block_to_tokens(block_id: int, block_size: int = 16, vocab=None):
    if vocab is None:
        vocab = [chr(i) for i in range(ord('b'), ord('z')+1)]
    rng = random.Random(block_id)
    tokens = [rng.choice(vocab) for _ in range(block_size)]
    return " ".join(tokens)

def process_json(jsonl_path: str, output_path: str):
    # 先统计总行数（为了 tqdm 显示进度）
    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing", unit="lines"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # 复制整个 item（里面已经有 chat_id, parent_chat_id 等等）
            new_item = dict(item)

            # 根据 hash_ids 生成 tokens
            ids = item["hash_ids"]
            tokens = [block_to_tokens(x) for x in ids]
            new_item["fake_tokens"] = " ".join(tokens)[:-16]

            # 每条 new_item 包含 meta 字段 + fake_tokens
            results.append(new_item)

    # 存 pickle，每个元素是一个完整的字典
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    process_json("qwen_traceA_blksz_16.jsonl", "processed.pkl")
