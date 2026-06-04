import string


def clean_text_file(file_path):
    try:
        # 1. 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_words = set()

        # 定义需要去除的标点符号（包含中英文标点）
        punctuation = string.punctuation + "，。！？、；：“”‘’（）【】《》"

        for line in lines:
            # 2. strip 空格，并去除指定的标点符号
            word = line.strip().translate(str.maketrans('', '', punctuation))

            # 如果去标点后不为空，则加入集合（自动去重）
            if word:
                cleaned_words.add(word)

        # 3. 将结果写回原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for word in sorted(cleaned_words, key=lambda x: lines.index(x) if x in lines else 0):
                f.write(word + '\n')

        # 4. 打印统计数量
        print(f"处理完成！")
        print(f"原文件已更新，共计保留了 {len(cleaned_words)} 个唯一词汇。")

    except FileNotFoundError:
        print("错误：找不到指定的文件，请检查路径。")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    # 请将此处替换为你的 TXT 文件路径
    target_file = r"/PythonProject/Fineweb/keyword_list/chemistry_keywords_EN.txt"
    clean_text_file(target_file)