import json
import argparse
import re

def compute_avg_first_chunk_time(file_path: str):
    total_time = 0.0
    count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # 预处理，把"}{"改成"}\n{"，保证每行都是一个合法 JSON
        content = re.sub(r'}\s*{', '}\n{', content)

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # if else 判断：只处理有 first_chunk_time 的情况
                if "first_chunk_time" in obj:
                    total_time += obj["first_chunk_time"]
                    count += 1
                else:
                    # 可选：调试时看看是什么数据
                    # print("跳过:", obj.keys())
                    pass
            except json.JSONDecodeError:
                # 出现坏行直接跳过
                continue

    if count == 0:
        print("⚠️ 文件里没有找到任何 first_chunk_time 字段")
        return

    avg_time = total_time / count
    print(f"✅ first_chunk_time 平均值: {avg_time:.6f} （基于 {count} 条记录）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算 JSON 文件中 first_chunk_time 的平均值")
    parser.add_argument("--file", 
        default="/home/shenyang/tests/burst/BurstGPT/example/profile_server/result/3000_arc_1_decode1.json", 
        type=str, help="输入 JSON 文件路径")
    args = parser.parse_args()

    compute_avg_first_chunk_time(args.file)
