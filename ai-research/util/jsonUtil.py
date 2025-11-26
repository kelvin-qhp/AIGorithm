import json
import pickle

def saveJson(filePath,data):
    with open(filePath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def readJson(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        return json.load(f)

def saveJson2(filePath,data):
    with open(filePath, 'wb') as f:
        pickle.dump(data, f)

def readJson2(filePath):
    with open(filePath, 'rb') as f:
        return pickle.load(f)