import json
import pickle

def load_dict(path):
    """
    加载字典
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_data(path):
    """
    读取txt文件, 加载训练数据
    :param path:
    :return:
    [{'text': ['当', '希', '望', ...],
     'label': ... }, {...}, ... ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return [eval(i) for i in f.readlines()]

def load_data(file_path,max_row_no=None):
    data = []
    with open(file_path, 'r',encoding='utf-8') as f:
        row_no = 0;
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按制表符分割，处理可能的空字段
            parts = line.split('\t')
            # 将数值字段转为 int/float，保留字符串特征
            row = []
            for p in parts:
                p = p.strip()
                if p == '':
                    row.append(0)  # 空值填充
                else:
                    try:
                        row.append(int(p))
                    except ValueError:
                        try:
                            row.append(int(p,16))
                        except ValueError:
                            row.append(p)  # 保留字符串（如哈希特征）
            data.append(row)

            if max_row_no is not None:
                row_no +=1
                if row_no >= max_row_no:
                    break

    return data

def write_data(path,data):
    with open(path,'wb+',encoding='utf-8') as f:
        pickle.dump(f)

def load_data2(path):
    with open(path,'rb',encoding='utf-8') as f:
        pickle.load(f)