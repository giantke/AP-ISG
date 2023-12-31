import json
from collections import Counter


def count_phrases(data, counter):
    """
    递归地遍历JSON数据，统计所有字符串值的出现频率。
    """
    if isinstance(data, dict):
        for value in data.values():
            count_phrases(value, counter)
    elif isinstance(data, list):
        for item in data:
            count_phrases(item, counter)
    elif isinstance(data, str):
        counter[data] += 1

def process_json_file(file_path):
    """
    处理JSON文件，统计并输出短语的出现频率。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        phrase_counter = Counter()
        count_phrases(data, phrase_counter)

        # 按频率排序
        sorted_phrases = sorted(phrase_counter.items(), key=lambda x: x[1], reverse=True)

        # 将结果保存到文件
        with open('phrase_frequency_result.txt', 'w', encoding='utf-8') as out_file:
            for phrase, count in sorted_phrases:
                out_file.write(f'"{phrase}": {count}\n')

        print("处理完成，结果已保存到output.txt")
    except Exception as e:
        print(f"发生错误: {e}")

# 调用函数处理你的JSON文件
process_json_file('/mnt/wsl/PHYSICALDRIVE3p1/chest-imagenome-dataset-1.0.0/silver_dataset/study_level_attribute_rdfgraphs.json')




