import json
import random
import re
from datasets import load_dataset


def load_keywords_from_files(file_list):
    """从多个txt文件中读取关键词，保持列表格式"""
    keywords = set()
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    keyword = line.strip().lower()
                    if keyword:
                        keywords.add(keyword)
        except FileNotFoundError:
            print(f"警告: 文件 {file_path} 未找到，已跳过。")
    return list(keywords)


def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def check_chemistry(text, keywords):
    text_lower = text.lower()
    matched_set = set()

    if is_chinese(text):
        for kw in keywords:
            if kw in text_lower:
                matched_set.add(kw)
    else:
        words = set(re.findall(r'\b[\w-]+\b', text_lower))
        for kw in keywords:
            if ' ' in kw:
                if kw in text_lower:
                    matched_set.add(kw)
            elif kw in words:
                matched_set.add(kw)
    return list(matched_set)


def process_data(keyword_files, output_file="result_1.json"):
    keywords = load_keywords_from_files(keyword_files)
    print(f"共加载了 {len(keywords)} 个关键词。")

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    chemistry_results = []
    not_chemistry_pool = []  # 用于存储所有 is_chemistry=0 的数据

    found_chemistry_count = 0

    print("开始遍历筛选，直到找到 50 条化学文章...")

    for example in dataset:
        text = example.get("text", "")
        matches = check_chemistry(text, keywords)
        keyword_count = len(matches)

        if keyword_count > 15:
            # 是化学数据
            record = {
                "text": text,
                "id": example.get("id", "N/A"),
                "url": example.get("url", "N/A"),
                "is_chemistry": 1,
                "keyword_count": keyword_count,
                "tag": ",".join(matches)
            }
            chemistry_results.append(record)
            found_chemistry_count += 1
            print(f"[{found_chemistry_count}/50] 收集到化学文章。")
        else:
            # 不是化学数据，存入池中
            record = {
                "text": text,
                "id": example.get("id", "N/A"),
                "url": example.get("url", "N/A"),
                "is_chemistry": 0,
                "keyword_count": keyword_count
            }
            not_chemistry_pool.append(record)

        # 核心停止条件：找到 50 条化学数据后停止
        if found_chemistry_count >= 50:
            break

    # 1. 保存 chemistry 数据
    with open("result_is_chemistry.json", 'w', encoding='utf-8') as f:
        json.dump(chemistry_results, f, ensure_ascii=False, indent=4)
    print(f"result_is_chemistry.json 保存完成。")



if __name__ == "__main__":
    txt_files = [
        "periodic_table_EN.txt",
        "periodic_table_CN.txt",
        "chemistry_keywords_EN.txt",
        "chemistry_keywords_CN.txt"
    ]
    process_data(txt_files)