from util.ExcelUtil import ExcelUtil
from collections import defaultdict

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
    FILE_PATH = "../data/input/gsol/Category_list.xlsx"
    df = ExcelUtil.read(fileName=FILE_PATH,sheetName="All")
    # df = df[df['category_id'] not in (50964,50971)]
    df = df[(~df['category_id'].isin([50964,50971]))]
    print(f'df size is:{df.shape}')
    # print(df.head())
    # df = pd.read_excel(FILE_PATH,index_col=0)
    all_cat = {row.get('category_id'):row for row in df.to_dict('records')}
    # print(all_cat)
    nodes = {}
    df_l4=df[df['category_level'] == 4]
    for r in df_l4.itertuples():
        print(r.category_id)
        cat_id,cat_name = retrieveTree(all_cat,r.category_id)
        # print(f'**** cat id:{cat_id}-{cat_name}')

        nodes[r.category_id] = {
            'cat_id': cat_id,
            'cat_name': cat_name
        }

        # if(len(cat_id))
        # p_r = all_cat.get(r.get('parent_category_id'))
        # (p_category_id,p_parent_category_id,p_category_level,p_desc_en) = p_r.category_id,p_r.parent_category_id,p_r.category_level,p_r.desc_en
        # print(f'******{p_category_id}-{p_parent_category_id}-{p_category_level}-p_desc_en')

    print(f"**** cat size:{len(nodes)} ")