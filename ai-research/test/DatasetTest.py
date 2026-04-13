from datasets import load_dataset
from  util.jsonUtil import readJson,readJson2
from util.ExcelUtil import ExcelUtil
import json
import pandas as pd
import chardet

from sklearn.model_selection import train_test_split


# def data_process(examples):
#     examples['l3_category_id'] =  [(all_categories.get(cat).get('cat_id'))[1] if all_categories.get(cat) is not None else 0 for cat in examples['category_id'] ]
#     examples['l3_cat_name'] =  [(all_categories.get(cat).get('cat_name'))[1] if all_categories.get(cat) is not None else '' for cat in examples['category_id'] ]
#
#     return examples


if __name__ == '__main__':

    #显示所有列
    pd.set_option('display.max_columns', None)
    #显示所有行
    pd.set_option('display.max_rows', None)
    #设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth',100)

    # all_categories = readJson2('categories_new.pkl')
    # print(f'categories json:{len(all_categories)}')
    #
    # # # df = ExcelUtil.readCsv("../data/input/product/PP_01.csv")
    # # # df_agg = df.groupby('category_id',as_index=False)['category_id'].agg({'cnt': pd.Series.count}).sort_values(by='cnt',ascending=True)
    # # # print(df_agg[:50])
    # #
    # # # my_dataset = load_dataset("csv",data_dir="../data/input/product/")
    # my_dataset = load_dataset("csv",data_dir="../data/input/product/",split="train")
    # # my_dataset = load_dataset("csv",data_files="../data/input/product/PP_01.csv",split="train")
    # print(my_dataset)
    #
    # my_dataset = my_dataset.map(data_process,batched=True,remove_columns=['org_id','short_description'])
    # my_dataset = my_dataset.filter(lambda x:x['l3_category_id'] != 0)
    # print(my_dataset)
    #
    # # print(my_dataset['category_id'][:5])
    # my_dataset = my_dataset.class_encode_column("l3_cat_name")
    # p_dataset = my_dataset.train_test_split(test_size=0.4,shuffle=True,stratify_by_column='l3_cat_name',seed=41)
    # print(f'split train-test dataset:{p_dataset}')
    #
    #
    # p2_dataset = p_dataset['test'].train_test_split(test_size=0.5,shuffle=True,stratify_by_column='l3_cat_name',seed=41)
    # print(f'split test-validation dataset:{p2_dataset}')
    #
    #
    # train_dataset = p_dataset['train'].with_format('pandas')
    #
    # train_dataset.remove_columns(['category_id','l3_cat_name'])
    #
    # df1,df2,df3,df4 = train_dataset[0:500000],train_dataset[500000:1000000],train_dataset[1000000:1500000],train_dataset[1500000:]
    # df1.to_csv("../data/input/gsol/train_1.csv",index=False,encoding='utf-8')
    # df2.to_csv("../data/input/gsol/train_2.csv",index=False,encoding='utf-8')
    # df3.to_csv("../data/input/gsol/train_3.csv",index=False,encoding='utf-8')
    # df4.to_csv("../data/input/gsol/train_4.csv",index=False,encoding='utf-8')
    #
    # p2_dataset['train'].to_csv("../data/input/gsol/validation.csv",index=False,encoding='utf-8')
    # p2_dataset['test'].to_csv("../data/input/gsol/test.csv",index=False,encoding='utf-8')

    # train_test_split()
    # all_categories = readJson2('categories_new.pkl')
    # print(f'categories json:{len(all_categories)}')
    # category_tree = all_categories.get(10008)
    # print(category_tree)
    #
    # filter_dataset = my_dataset.filter(lambda example:all_categories.get(example['category_id']) is not None)
    # print(filter_dataset)

    # FILE_PATH = "../data/input/gsol/L3_Category_list2.csv"
    # df = ExcelUtil.readCsv(fileName=FILE_PATH,sheetName="All")
    # print(df.shape)

    # df.drop_duplicates(inplace=True)
    # print(df.shape)

    # result =df.groupby('keyword',as_index=False)['keyword'].agg({'cnt': pd.Series.count}).sort_values(by='cnt',ascending=False)
    # result2 = result[result['cnt']>1]
    # keywords = [k for k in result2['keyword']]
    # print(len(result2),keywords)

    # with open('../data/input/gsol/train_1.csv', 'rb') as f:
    #     result = chardet.detect(f.read())
    # print(result['encoding'])

    # df.to_csv("../data/input/gsol/L3_Category_list2.csv")
    all_files = ['../data/input/gsol2/train_1.csv','../data/input/gsol2/train_2.csv','../data/input/gsol2/train_3.csv','../data/input/gsol2/train_4.csv']
    df = pd.concat([pd.read_csv(file) for file in all_files])
    # df = pd.read_csv('../data/input/gsol2/train_4.csv',encoding='utf-8')
    all_cat_id = df['l3_category_id'].unique()
    print(len(all_cat_id))

    # df.drop(columns=['category_id','l3_cat_name'],inplace=True)
    # df.to_csv('../data/input/gsol2/validation.csv',encoding='utf-8',index=False)
    print(len(df['l3_category_id'].unique()))
