import json
import os
import re
import math
import time
import logging
import glob
import pyarrow.parquet as pq
from collections import Counter

# ================== 配置区域 ==================
DATASET_NAME = "CC-MAIN-2016-07"
BASE_PATH = r"/mnt/nas_new/chemistry/LJQ/fineweb_classified_2016"
OUTPUT_DIR = os.path.join(BASE_PATH, DATASET_NAME)
DATA_SOURCE_PATH = f"/mnt/nas_new/chemistry/LJQ/fineweb/{DATASET_NAME}/*.parquet"

WEIGHTS = {
    "chemistry_keyword_common_EN": 1.0,
    "chemistry_keyword_unique_EN": 3.0,
    "chemistry_keyword_ambiguous_EN": 0.2,
    "periodic_table_ambiguous_EN": 0.2,
    "periodic_table_rare_EN": 1.0,
    "periodic_table_unique_EN": 2.5,
}


# ================== 日志配置 ==================
def setup_logging():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    logger = logging.getLogger("processor")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 文件日志
    log_file = os.path.join(BASE_PATH, f"{DATASET_NAME}_log.txt")
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台日志
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


# ================== 分类器 ==================
class ChemicalClassifier:
    def __init__(self, key_dir):
        self.keyword_sets = {}
        key_list_dir = os.path.join(key_dir, "keyword_lists")
        for key in WEIGHTS.keys():
            path = os.path.join(key_list_dir, f"{key}.txt")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.keyword_sets[key] = {line.strip().lower() for line in f if line.strip()}

    def classify(self, text, alpha=0.1, beta=5, threshold=0.45):
        text_lower = text.lower()
        matched_set = set()
        words = Counter(re.findall(r'\b[\w-]+\b', text_lower))
        raw_score = 0
        for word, frequency in words.items():
            for key, keywords in self.keyword_sets.items():
                if word in keywords:
                    matched_set.add(word)
                    raw_score += WEIGHTS[key] * math.log(1 + frequency)
        norm_score = 1 / (1 + math.exp(-alpha * (raw_score - beta)))
        return norm_score > threshold, norm_score, list(matched_set)


# ================== 主程序 ==================
def process_data():
    logger = setup_logging()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    classifier = ChemicalClassifier(BASE_PATH)
    all_files = sorted(glob.glob(DATA_SOURCE_PATH))

    # Checkpoint: 读取已生成的json文件名，跳过已处理的文件
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
    processed_names = {f.rsplit('_', 1)[0] for f in existing_files}

    global_total_rows = 0
    global_total_matched = 0

    logger.info(f"🚀 发现 {len(all_files)} 个parquet文件，准备开始处理...")

    for file_path in all_files:
        file_name = os.path.basename(file_path).replace('.parquet', '')

        if file_name in processed_names:
            continue

        start_time = time.time()
        matched_records = []
        file_rows = 0

        # 分块处理：避免内存溢出，读取所有列
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=10000):
            df_chunk = batch.to_pandas()
            for _, row in df_chunk.iterrows():
                file_rows += 1

                # 放在 row 的遍历循环内
                file_rows += 1
                if file_rows % 50000 == 0:  # 每处理 5 万行打印一次
                    logger.info(f"正在处理 {file_name}，当前进度: {file_rows} 行...")

                text = row.get("text", "")
                is_chem, score, matches = classifier.classify(text)

                if is_chem:
                    # 获取完整行数据，并更新分类信息
                    record = row.to_dict()
                    record.update({
                        "classification_score": score,
                        "tag": ",".join(matches)
                    })
                    matched_records.append(record)

        # 保存结果
        match_count = len(matched_records)
        if match_count > 0:
            out_name = f"{file_name}_{match_count}.json"
            with open(os.path.join(OUTPUT_DIR, out_name), 'w', encoding='utf-8') as f:
                json.dump(matched_records, f, ensure_ascii=False, indent=4)

        global_total_rows += file_rows
        global_total_matched += match_count

        elapsed = time.time() - start_time
        logger.info(
            f"文件: {file_name}.parquet | 耗时: {elapsed:.2f}s | 本文件处理: {file_rows}行 | 命中: {match_count}条")

    logger.info(f"✅ 处理完毕。所有扫描文件总行数: {global_total_rows}，化学相关总行数: {global_total_matched}")


if __name__ == "__main__":
    process_data()