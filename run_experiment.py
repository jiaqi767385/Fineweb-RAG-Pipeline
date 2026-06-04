import json
import re

INPUT_FILE = 'RAG_results/fineweb_test_dataset.json'
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
    matches = list(matched_set)
    keyword_count = len(matches)
    return keyword_count > 40


def process_data(keyword_files):
    keywords = load_keywords_from_files(keyword_files)
    print(f"共加载了 {len(keywords)} 个关键词。")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到文件 {INPUT_FILE}")
        return

    print("开始遍历筛选")
    misjudged_chemistry_results = []
    count = 0
    for example in dataset:
        text = example.get("text", "")
        is_chemistry = check_chemistry(text, keywords)
        if is_chemistry == example.get("is_chemistry", 0):
            count += 1
        else:
            misjudged_chemistry_results.append(example)
    print(f"在当前判断标准下，精确度为{count / 1000}")
    with open("misjudged_chemistry_results.json", 'w', encoding='utf-8') as f:
        json.dump(misjudged_chemistry_results, f, ensure_ascii=False, indent=4)
    print(f"misjudged_chemistry_results.json 保存完成。")



if __name__ == "__main__":
    txt_files = [
        "periodic_table_EN.txt",
        "periodic_table_CN.txt",
        "chemistry_keywords_EN.txt",
        "chemistry_keywords_CN.txt"
    ]
    process_data(txt_files)