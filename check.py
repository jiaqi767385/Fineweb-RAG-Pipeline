import json
import random
import time

import requests

# --- 配置部分 ---
BASE_URL = "http://10.200.95.16:30300/v1"
API_KEY = "sk-I8NANWiy8lYo64xWdzVF2xFfK00fNUu0wQSiqtIPxgugh4B6"
MODEL_NAME = "DeepSeek-V4-flash-zj"
INPUT_FILE = "RAG_results/fineweb_test_dataset.json"
OUTPUT_FILE = "fineweb_test_dataset_misjudged.json"

sample_size = 5


def is_chemistry_by_llm(text):
    """通过 API 调用模型判断文本是否为化学相关 (返回 1 或 0)"""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
    你现在是一位专业的科学文献审核员。请判断提供的文本中是否包含化学相关的知识内容。

你的判定准则（必须严格执行）：
1.【忽略主题】不要判断文本的核心主题（如：艺术、历史、考古、农业等）。即使化学知识只是文章的背景介绍，也属于判定范围。
2.【化学触发项】只要文中出现了以下任何一种内容，即判定为“1”：
   - 化学物质：具体的元素名称、化合物名称、分子式、离子、有机/无机物。
   - 化学过程：化学反应、电解、合成、氧化、还原、催化、钝化、生物代谢的化学描述。
   - 化学理论与概念：原子结构、元素周期表、周期律、同位素、量子化学、价键、轨道、摩尔、度量衡的化学溯源。
   - 化学工业：工业提纯、化工副产品、材料合成、炼油/炼丹工艺。
3.【判定逻辑】只要文本中存在上述任意要素，直接回复 1；如果不包含任何化学相关信息，才回复 0。

输出限制：
- 只允许输出一个字符：1 或 0。
- 严禁输出任何解释、分析、符号或标点。

文本内容：
{text}
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        # 如果遇到 503，尝试休眠更久再返回
        if response.status_code == 503:
            print("警告: 服务器繁忙 (503)，休眠 10 秒后继续...")
            time.sleep(10)
            return 0
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        return 1 if '1' in content else 0
    except Exception as e:
        print(f"API 调用异常: {e}")
        return 0


def main():
    # 1. 读取数据
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到文件 {INPUT_FILE}")
        return

    if len(data) > sample_size:
        print(f"原始数据集大小为 {len(data)}，现随机抽取 {sample_size} 条进行验证...")
        data_to_verify = random.sample(data, sample_size)
    else:
        print(f"数据集小于 {sample_size} 条，将验证所有数据。")
        data_to_verify = data

    print(f"开始进行 LLM 二次验证...")

    # 2. 验证与统计
    misjudged_data = []
    total = len(data_to_verify)  # 修改：基于抽取后的数量

    for i, item in enumerate(data_to_verify):
        text = item.get("text", "")
        # 调用 LLM 判断
        if is_chemistry_by_llm(text) != item.get("is_chemistry", 0):
            misjudged_data.append(item)
            print(f"[{i + 1}/{total}] 发现错判")
        else:
            print(f"[{i + 1}/{total}] 确认正确")

        # 频率限制
        time.sleep(0.5)

    # 3. 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(misjudged_data, f, ensure_ascii=False, indent=4)

    # 4. 打印准确率
    err_count = len(misjudged_data)
    accuracy = ((total - err_count) / total) * 100 if total > 0 else 0

    print("-" * 30)
    print(f"抽样验证完成！")
    print(f"总检查数: {total}")
    print(f"算法误判数 : {err_count}")
    print(f"在该样本中的准确率: {accuracy:.2f}%")
    print(f"误判数据已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
