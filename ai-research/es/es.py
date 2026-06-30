from elasticsearch import Elasticsearch,helpers
from embedding.mpnet_embedding import sentence_transformers_embedding

import time


def es_client():
    es = Elasticsearch(
        ['https://192.168.117.26:9200'],
        verify_certs=False,  # 不验证证书
        ssl_show_warn=False,
        basic_auth=('elastic', 'za8yNrYIq*rrm_We1UR6')
    )
    if es.ping():
        print("✅ 成功连接到 Elasticsearch")
    else:
        print("❌ 连接失败")

    return es

es = es_client()

# 创建索引（带向量字段）
def create_indices(indices_name="pp_vector"):
    """创建带有向量字段的索引"""

    mapping = {
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "product_name": {"type": "text", "analyzer": "standard"},
                "product_vector": {
                    "type": "dense_vector",
                    "dims": 768,  # 向量维度
                    "index": True,
                    "similarity": "cosine"  # 相似度计算方式
                },
                "org_id":{"type": "long"},
                "l4_category_id":{"type": "integer"},
                "l4_category_name":{"type": "text"}
            }
        },
        "settings": {
            "number_of_shards": 1,  # 单分片
            "number_of_replicas": 0  # 无副本
        }
    }

    # 删除已存在的索引（如果存在）
    if es.indices.exists(index=indices_name):
        es.indices.delete(index=indices_name)
        print(f"🗑️  删除已存在的索引: {indices_name}")

    # 创建索引
    es.indices.create(index=indices_name, body=mapping)
    print(f"✅ 创建索引: {indices_name}")


def insert(indices_name, pp_id, pp_name):
    """写入单条文档"""
    pp_vector = sentence_transformers_embedding(pp_name)
    doc = {
        "product_id": pp_id,
        "product_name": pp_name,
        "product_vector": pp_vector
    }

    response = es.index(
        index=indices_name,
        id=pp_id,
        body=doc
    )

    print(f"✅ 写入文档: {pp_id}")
    return response


def batch_insert( documents, indices_name="pp_vector",batch_size=100):
    """
    批量写入向量文档

    Args:
        indices_name: 索引名称
        documents: 文档列表，每个文档包含 id, text, metadata
        batch_size: 批次大小
    """

    # 准备批量数据
    actions = []

    product_names = [d["product_name"] for d in documents]
    pp_vectors = sentence_transformers_embedding(product_names)

    for i, doc in enumerate(documents):
        # 生成向量


        # 构建文档
        action = {
            "_index": indices_name,
            "_id": doc['product_id'],
            "_source": {
                "product_id": doc['product_id'],
                "product_name": doc['product_name'],
                "product_vector": pp_vectors[i]
            }
        }
        actions.append(action)

        # 批量写入
        if len(actions) >= batch_size:
            success, failed = helpers.bulk(es, actions, stats_only=True)
            print(f"✅ 批量写入: {success} 成功, {failed} 失败")
            actions = []

    # 写入剩余数据
    if actions:
        success, failed = helpers.bulk(es, actions, stats_only=True)
        print(f"✅ 批量写入: {success} 成功, {failed} 失败")

    print(f"📊 总共写入: {len(documents)} 条文档")

def batch_insert2(df, indices_name="pp_vector"):
    """
    批量写入向量文档

    Args:
        indices_name: 索引名称
        documents: 文档列表，每个文档包含 id, text, metadata
        batch_size: 批次大小
    """

    df['_index'] = indices_name
    df['_id'] = df['product_id']

    # 批量写入
    success, failed = helpers.bulk(es, df.to_dict("records"), stats_only=True)
    print(f"✅ 批量写入: {success} 成功, {failed} 失败")


    print(f"📊 总共写入: {len(df)} 条文档")

def knn_search(indices_name, query_text, k=5, num_candidates=100):
    """
    基础 kNN 搜索

    Args:
        indices_name: 索引名称
        query_text: 查询文本
        k: 返回结果数量
        num_candidates: 候选数量（越大越准确，但越慢）
    """
    if k >2000:
        k=2000

    if num_candidates < k:
        num_candidates = k * 2

    if num_candidates >= 2000:
        num_candidates = 2000

    # 生成查询向量
    start = time.time()
    query_vector = sentence_transformers_embedding(query_text).tolist()
    print(f"embedding耗时: {time.time() - start:.4f} 秒")
    start = time.time()
    # kNN 查询
    response = es.search(
        index=indices_name,
        body={
            "size": k,
            "knn": {
                "field": "product_vector",  # 向量字段名
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates
            }
        }
    )
    print(f"query耗时: {time.time() - start:.4f} 秒")
    return response['hits']['hits']

if __name__ == '__main__':
    # es = es_client()
    # create_indices()
    # insert("pp_vector", "123", "python")
    # documents = [{"product_id": "1201307509", "product_name": "Promotional Mother's Day Balloons Heart Shape Custom Color Decorative Foil Balloons"},
    #              {"product_id": "1164142860", "product_name": "HDMI Matrix 4x4, with UTP Extender 60m"}]
    # batch_insert(documents)
    # bearing
    start = time.time()
    results = knn_search("pp_vector", "genuine leather shoes",800)
    print(f"Total耗时: {time.time() - start:.4f} 秒 for size:{len(results)}")

    for hit in results:
        print(f"得分: {hit['_score']:.4f} id:{hit['_id']} 文本: {hit['_source']['product_name']}")
        print("-" * 50)

