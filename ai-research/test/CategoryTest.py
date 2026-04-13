from util.ExcelUtil import ExcelUtil
from collections import defaultdict
import json
from util.jsonUtil import readJson2,saveJson2,saveJson
import pandas as pd


def _build_tree(self, data):
    """构建树形结构的核心方法"""
    # 创建节点映射
    nodes = {}
    children_map = defaultdict(list)

    # 第一遍：收集所有节点和父子关系
    for item in data:
        category_id = item.get('category_id')
        parent_id = item.get('parent_id')

        nodes[category_id] = {
            'id': category_id,
            'name': item.get('category_name', ''),
            'data': item,  # 保存原始数据
            'children': []
        }

        if parent_id and parent_id in nodes:
            children_map[parent_id].append(category_id)

    # 第二遍：构建树形结构
    root_nodes = []
    for category_id, node in nodes.items():
        parent_id = node['data'].get('parent_id')

        # 添加子节点
        for child_id in children_map[category_id]:
            if child_id in nodes:
                node['children'].append(nodes[child_id])

        # 如果是根节点
        if not parent_id or parent_id not in nodes:
            root_nodes.append(node)

    self.tree_dict = root_nodes
    self._create_flat_dict()

    return root_nodes

def retrieveTree(all_cat,c_id):
    _c = all_cat.get(c_id)
    _c_pd_id = _c.get('parent_category_id')
    _p = all_cat.get(_c_pd_id)
    if _p is None:
        return (_c.get('category_id'),),(_c.get('desc_en'),)
    elif _p.get('category_level') == 1:
        return (_c.get('category_id'),_p.get('category_id')),(_c.get('desc_en'),_p.get('desc_en'))
    else:
        (_p_cat_id,_p_cat_name) = retrieveTree(all_cat,_c_pd_id)
        return ((_c.get('category_id'),)+_p_cat_id),((_c.get('desc_en'),)+_p_cat_name)

if __name__ == '__main__':
    FILE_PATH = "../data/input/gsol/org_cat_list.csv"
    # # df = ExcelUtil.read(fileName=FILE_PATH,sheetName="All")
    # df = ExcelUtil.readCsv(fileName=FILE_PATH,sheetName="All")
    # # df = df[df['category_id'] not in (50964,50971)]
    # # df = df[(~df['category_id'].isin([50964,50971]))]
    # print(f'df size is:{df.shape}')
    # # print(df.head())
    # # df = pd.read_excel(FILE_PATH,index_col=0)
    # all_cat = {row.get('category_id'):row for row in df.to_dict('records')}
    # # print(all_cat)
    # nodes = {}
    # df_l4=df[df['category_level'] == 4]
    # for r in df_l4.itertuples():
    #     print(r.category_id)
    #     # filter a special L4 cat id without any parent root l1/L2/l3
    #     if r.category_id == 8004:
    #         continue
    #
    #     cat_id,cat_name = retrieveTree(all_cat,r.category_id)
    #     # print(f'**** cat id:{cat_id}-{cat_name}')
    #
    #     nodes[r.category_id] = {
    #         'cat_id': cat_id,
    #         'cat_name': cat_name
    #     }
    #
    # print(f"**** cat size:{len(nodes)} ")
    # saveJson2('categories_new.pkl',nodes)
    # saveJson('categories.json',nodes)


    # data = readJson2('categories_new.pkl')
    # print(f'categories json:{len(data)}')
    # category_tree = data.get(10008)
    # print(category_tree.get('cat_id'))
    #
    # cat_pp_df = ExcelUtil.readCsv("../data/input/gsol/cat_pp_paid_agg.csv")
    # # cat_pp_df2 = cat_pp_df[cat_pp_df['category_id']==8004]
    # # cat_pp_df['l1_cat_id'] = cat_pp_df['category_id'].apply(lambda col: len(data.get(col).get('cat_id')))
    # cat_pp_df['l1_cat_id'] = cat_pp_df['category_id'].apply(lambda col: (data.get(col).get('cat_id'))[3])
    # cat_pp_df['l1_cat_name'] = cat_pp_df['category_id'].apply(lambda col: (data.get(col).get('cat_name'))[3])
    # cat_pp_df['l2_cat_id'] = cat_pp_df['category_id'].apply(lambda col: (data.get(col).get('cat_id'))[2])
    # cat_pp_df['l2_cat_name'] = cat_pp_df['category_id'].apply(lambda col: (data.get(col).get('cat_name'))[2])
    # cat_pp_df['l3_cat_id'] = cat_pp_df['category_id'].apply(lambda col: (data.get(col).get('cat_id'))[1])
    # cat_pp_df['l3_cat_name'] = cat_pp_df['category_id'].apply(lambda col: (data.get(col).get('cat_name'))[1])
    # cat_pp_df.to_csv('cat_pp_paid_agg_output.csv')
    # print(cat_pp_df.info)

    # df = pd.read_csv(FILE_PATH)
    # df = df[df['category_level'] == 3]
    # all_cat = {row.get('category_id'):row.get('desc_en') for row in df.to_dict('records')}
    # print(f'***:{len(all_cat)}')
    # saveJson2('../data/input/gsol/l3_categories.pkl',all_cat)

    # l3_categories = readJson2('../data/input/gsol/l3_categories.pkl')
    # print(f'categories json:{len(l3_categories)}')
    # all_l3 = [k for k,v in l3_categories.items()]
    # print(len(list(all_l3)))
    #
    # all_files = ['../data/input/gsol2/train_1.csv','../data/input/gsol2/train_2.csv','../data/input/gsol2/train_3.csv','../data/input/gsol2/train_4.csv']
    # df = pd.concat([pd.read_csv(file) for file in all_files])
    # all_cat_id = df['l3_category_id'].unique()
    # print(len(all_cat_id))
    # l3_cat_id = df['l3_category_id'].unique()
    # print(f' l3 cat id count:{len(l3_cat_id)}')
    # diff_id = [a for a in all_l3 if a not in l3_cat_id]
    # print(f'{len(diff_id)} cat id:{diff_id} lack of PP')
    #
    # filte_l3 = {int(cat):l3_categories.get(cat) for cat in all_cat_id}
    # print(f'final cat lenght:{len(filte_l3)}')
    # saveJson2('../data/input/gsol/l3_categories_all.pkl',filte_l3)

    l3_categories = readJson2('../data/input/gsol/l3_categories_all.pkl')
    print(f'categories json:{(l3_categories)}')

