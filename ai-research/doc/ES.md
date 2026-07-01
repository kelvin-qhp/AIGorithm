# ES

### -2. Troubleshooting high CPU 
https://www.zyxy.net/archives/38101
~~~
1. top -H -pid xxx

十制转十六制度
2. printf "%x\n" {pid}

3. jstack {十六进制pid} dump1.txt
-> RUNNABLE 
-> WAITTING
-> BLOCKED

4. grep RUNNABLE dump1.txt | wc -l


~~~


### -1. Misc
~~~
Python Study:
https://liaoxuefeng.com/books/python/async-io/asyncio/index.html#0
~~~

### 0. GSOL isearch biz:

~~~
1.1 isearch PP search list cache key:
topK: isearch.s.topK.{power}.MANUAL_ADV_SAB.DESKTOP.en.01F0DAE50E8C742405C73F464208747C
agg: isearch.s.agg.{power}.MANUAL_ADV_SAB.DESKTOP.en.44F808F50EAB7AC6B3B942EDC70DB20A
1.2 isearch SP search list 
agg:isearch.s.agg.{power bank}.DESKTOP.en.7EE855A2F99F162A2E5535626533C099

2. Show genie:
SP list: isearch.agg.v1.ts.Supplier.agg.filter.[2803000000044, 2803000000046, 2803000000372]edb9f5ca46053014f012293b0f9a1720

3。 HK exhibitor:
SP list: isearch.agg.hk.show.filters.A6B045BC507EFC86F37122CBEBB0A960
~~~

~~~
Popup:
1. receive message from kafka:
kubernetes.container_name.keyword:("gsol-dw-base")  and log:("[Popup-Collect] convert2Map from kafka for anonymous_id" )

2. filter unused message from kafka:
kubernetes.container_name.keyword:("gsol-dw-base")  and log:("[Popup-Collect] filterMessage" )

3. sync DB:
kubernetes.container_name.keyword:("gsol-dw-base")  and log:("[Popup-Collect] Sync2DB tableNo" )

4。 resend kafka:
kubernetes.container_name.keyword:("gsol-dw-base")  and log:("[Popup-Collect] Send2Kafka topic:" )

~~~



### 1. Data too large Error

~~~
PUT _cluster/settings
{
  "persistent" : {
    "indices.breaker.fielddata.limit" : "40%" 
  }
}


indices.fielddata.cache.size:  40%

indices.fielddata.cache.size:
(Static) The max size of the field data cache, eg 38% of node heap space, or an absolute value, eg 12GB. Defaults to unbounded. If you choose to set it, it should be smaller than Field data circuit breaker limit.

indices.breaker.total.use_real_memory:
(Static) Determines whether the parent breaker should take real memory usage into account (true) or only consider the amount that is reserved by child circuit breakers (false). Defaults to true.

indices.breaker.total.limit :
(Dynamic) Starting limit for overall parent breaker. Defaults to 70% of JVM heap if indices.breaker.total.use_real_memory is false. If indices.breaker.total.use_real_memory is true, defaults to 95% of the JVM heap.

GET /_cat/fielddata
GET /_cat/fielddata?v=true
GET /_stats/fielddata?fields=*

GET _nodes/stats/breaker?
GET _cluster/stats			--view /docs/storecache/fielddata/Nodes(CPU/Mem)
GET _cluster/settings?flat_settings=true

POST spu/_cache/clear
POST _cache/clear

GET /_stats/fielddata?fields=*
GET /_nodes/stats/indices/fielddata?fields=*
GET /_nodes/stats/indices/fielddata?level=indices&fields=*


get /_cat/thread_pool?v
~~~



~~~
在elasticsearch.yml 中设置boostrap.mlockall属性为true；

设置Xmx和Xms属性值相同，避免JVM改变堆大小，注意内存大小合适，并非越大越好(会导致回收时长变长)；

修改/etc/security/limits.conf，添加如下内容(假设运行ES的用户是appuser）
appuser - nofile 65536
appuser - memlock unlimited

修改/etc/pam.d/common-session文件，添加如下内容
session required pam_limits.so

~~~

auto_queue_frame_size ：

![1773133730205](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1773133730205.png)



### 3. doccano install 

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

~~~
environs
furl
whitenoise
django-filter
polymorphic
django-polymorphic
django-cors-headers
django-allauth
dj_rest_auth
django_celery_results
django_drf_filepond
psycopg2
auto_labeling_pipeline
djangorestframework
url
filetype
chardet
pyexcel
seqval
pandas
django-rest-polymorphic
model_mommy
djangorestframework-xml

MIDDLEWARE:
allauth.account.middleware.AccountMiddleware



~~~



### Add indice aliases:

```
POST /_aliases
{
    "actions" : [
        { "add" : { "index" : "spu_20230411141951", "alias" : "spu" } }
    ]
}

POST /_aliases
{
    "actions" : [
        { "remove" : { "index" : "spu_20230411141951", "alias" : "spu" } }
    ]
}
```

### delete by query

~~~
POST /spu_fr/_delete_by_query
{
  "query": {
    "terms": {
      "supplierLevel": [
        "-2"
      ]
    }
  }
}
~~~

### update  field value by id

~~~

POST /keyword_dict/_doc/746/_update11
{
  "doc": {
    "searchMetric":0.986
  }
}
~~~



### Suggest for product

~~~
POST /keyword/_search
{
  "suggest" : {
    "suggestion" : {
      "prefix" : "po",
      "completion" : {
        "field" : "suggest",
        "size" : 5,
        "contexts" : {
          "searchType" : [ {
            "context" : "Product",
            "boost" : 1,
            "prefix" : false
          } ]
        }
      }
    }
  }
}
~~~

### Suggest for Supplier

~~~
POST /keyword/_search
{
  "from" : 0,
  "size" : 5,
  "query" : {
    "bool" : {
      "must" : [ {
        "match" : {
          "keyword" : {
            "query" : "li",
            "operator" : "OR",
            "prefix_length" : 0,
            "max_expansions" : 50,
            "fuzzy_transpositions" : true,
            "lenient" : false,
            "zero_terms_query" : "NONE",
            "auto_generate_synonyms_phrase_query" : true,
            "boost" : 1.0
          }
        }
      } ],
      "filter" : [ {
        "bool" : {
          "must" : [ {
            "term" : {
              "searchType" : {
                "value" : "Supplier",
                "boost" : 1.0
              }
            }
          } ],
          "adjust_pure_negative" : true,
          "boost" : 1.0
        }
      } ],
      "adjust_pure_negative" : true,
      "boost" : 1.0
    }
  },
  "_source" : {
    "includes" : [ "keyword" ],
    "excludes" : [ ]
  },
  "highlight" : {
    "pre_tags" : [ "<strong>" ],
    "post_tags" : [ "</strong>" ],
    "fields" : {
      "keyword" : { }
    }
  }
}
~~~

### ES Tokenizer

~~~
NOTES: 
tokenizer = 'Stanadrd' will remove special sign such as ', -, _
 		= 'whitespace' only split them into term with " "
 		
GET /_analyze
{
  "analyzer":"simple",
  "text":"Brown_Foxes jumped over the lazy dog's bone"
}

GET /_analyze
{
  "tokenizer":"edge_ngram",
  "text":"Brown_Foxes jumped over the lazy dog's bone"
}

GET /_analyze
{
    "tokenizer": {
          "type": "edge_ngram",
          "min_gram": 2,
          "max_gram": 3,
          "token_chars": [
            "letter",
            "digit"
          ]
        
      },
"text":"Brown_Foxes jumped over the lazy dog's bone"
}

POST _analyze
{
 "tokenizer": "keyword",
  "filter": [ "lowercase" ],
  "text": "john.SMITH@example.COM"
}


POST _analyze
{
  "tokenizer": "standard",
  "filter": [
    {
      "type": "dictionary_decompounder",
      "word_list": ["power", "bank", "meer", "ah"]
    }
  ],
  "text": "powerbank 1200ah tah powercm"
}

POST _analyze
{
  "tokenizer": {
    "type": "simple_pattern_split",
    "pattern": ","
  },
  "filter": [
    "lowercase"
  ],
  "text": "Powerbank,1200ah"
}
~~~

### Suggestion for Supplier

~~~

POST /keyword_20250811173636/_search
{
  "from": 0,
  "size": 5,
  "query": {
    "bool": {
      "must": [
        {
          "bool": {
            "should": [
              {
                "match": {
                  "keyword": {
                    "query": "dongguan bluex",
                    "operator": "OR",
                    "prefix_length": 0,
                    "max_expansions": 50,
                    "fuzzy_transpositions": true,
                    "minimum_should_match": "100%",
                    "lenient": false,
                    "zero_terms_query": "NONE",
                    "auto_generate_synonyms_phrase_query": true,
                    "boost": 1
                  }
                }
              },
              {
                "wildcard": {
                  "keyword.raw": {
                    "value": "Dongguan blueking*"
                  }
                }
              }
            ],
            "adjust_pure_negative": true,
            "minimum_should_match": "1",
            "boost": 1
          }
        }
      ],
      "filter": [
        {
          "bool": {
            "must": [
              {
                "term": {
                  "searchType": {
                    "value": "Supplier",
                    "boost": 1
                  }
                }
              }
            ],
            "adjust_pure_negative": true,
            "boost": 1
          }
        }
      ],
      "adjust_pure_negative": true,
      "boost": 1
    }
  },
  "_source": {
    "includes": [
      "keyword"
    ],
    "excludes": []
  },
  "highlight": {
    "pre_tags": [
      "<strong>"
    ],
    "post_tags": [
      "</strong>"
    ],
    "fields": {
      "keyword": {}
    }
  }
}
~~~



### rank feature

~~~

PUT my-index-000001
{
  "mappings": {
    "properties": {
      "pagerank": {
        "type": "rank_feature"
      },
      "url_length": {
        "type": "rank_feature",
        "positive_score_impact": false
      },
      "topics": {
        "type": "rank_features"
      }
    }
}

}

PUT my-index-000001/_doc/1
{
  "url": "https://en.wikipedia.org/wiki/2016_Summer_Olympics",
  "content": "Rio 2016",
  "pagerank": 10,
  "url_length": 42,
  "topics": {
    "sports": 50,
    "brazil": 30
  }
}

PUT my-index-000001/_doc/2
{
  "url": "https://en.wikipedia.org/wiki/2016_Brazilian_Grand_Prix",
  "content": "Formula One motor race held on 13 November 2016",
  "pagerank": 50,
  "url_length": 47,
  "topics": {
    "sports": 35,
    "formula one": 65,
    "brazil": 20
  }
}

PUT my-index-000001/_doc/3
{
  "url": "https://en.wikipedia.org/wiki/Deadpool_(film)",
  "content": "Deadpool is a 2016 American superhero film",
  "pagerank": 90,
  "url_length": 37,
  "topics": {
    "movies": 60,
    "super hero": 65
  }
}


GET my-index-000001/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "2016"
          }
        }
      ],
      "should": [
        {
          "rank_feature": {
            "field": "pagerank"
          }
        },
        {
          "rank_feature": {
            "field": "url_length",
            "boost": 1
          }
        }
      ]
    }
  },
  "explain": true
}

~~~

### GET /_cat/tasks?v

--GET /_cat/tasks?v

--GET _tasks?detailed=true
--GET _tasks?&detailed&actions=indices:data/read/search&nodes=hPDJUkPUQNOLDRvLkMuDZQ
--GET _tasks?actions=*&detailed=true&nodes=hPDJUkPUQNOLDRvLkMuDZQ
--POST _tasks/Zx2pLsOIRgOd08IBVlZS4g:5982548470/_cancel

--GET _tasks/E_IDzeBLSBC0Ils0aJEFsA:36967411

--get _nodes/hot_threads

![1770886036528](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1770886036528.png)

~~~
action                                   task_id                           parent_task_id                    type       start_time    timestamp running_time ip           node
data_frame/transforms[c]                 E_IDzeBLSBC0Ils0aJEFsA:183182     cluster:75                        persistent 1747217489158 10:11:29  95.7d        10.38.56.198 es-master
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5898004721 -                                 transport  1755478573141 00:56:13  2.1h         10.38.56.198 es-master
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789000626 E_IDzeBLSBC0Ils0aJEFsA:5898004721 transport  1755478573156 00:56:13  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789000625 E_IDzeBLSBC0Ils0aJEFsA:5898004721 transport  1755478573156 00:56:13  2.1h         10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5789014258 -                                 transport  1755478581141 00:56:21  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789014434 hPDJUkPUQNOLDRvLkMuDZQ:5789014258 direct     1755478581209 00:56:21  2.1h         10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5789017942 -                                 transport  1755478583593 00:56:23  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789017956 hPDJUkPUQNOLDRvLkMuDZQ:5789017942 direct     1755478583602 00:56:23  2.1h         10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5982538574 -                                 transport  1755478588138 00:56:28  2.1h         10.38.79.31  es-data2
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789023879 Zx2pLsOIRgOd08IBVlZS4g:5982538574 transport  1755478588161 00:56:28  2.1h         10.38.49.21  es-data1
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5898030756 -                                 transport  1755478591023 00:56:31  2.1h         10.38.56.198 es-master
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789027992 E_IDzeBLSBC0Ils0aJEFsA:5898030756 transport  1755478591077 00:56:31  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789027991 E_IDzeBLSBC0Ils0aJEFsA:5898030756 transport  1755478591077 00:56:31  2.1h         10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5982548470 -                                 transport  1755478593846 00:56:33  2.1h         10.38.79.31  es-data2
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789031400 Zx2pLsOIRgOd08IBVlZS4g:5982548470 transport  1755478593858 00:56:33  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789031401 Zx2pLsOIRgOd08IBVlZS4g:5982548470 transport  1755478593858 00:56:33  2.1h         10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5789034478 -                                 transport  1755478596138 00:56:36  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789034497 hPDJUkPUQNOLDRvLkMuDZQ:5789034478 direct     1755478596150 00:56:36  2.1h         10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5982557997 -                                 transport  1755478599024 00:56:39  2.1h         10.38.79.31  es-data2
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789037913 Zx2pLsOIRgOd08IBVlZS4g:5982557997 transport  1755478599034 00:56:39  2.1h         10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5789041103 -                                 transport  1755478601855 00:56:41  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789041108 hPDJUkPUQNOLDRvLkMuDZQ:5789041103 direct     1755478601873 00:56:41  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789041109 hPDJUkPUQNOLDRvLkMuDZQ:5789041103 direct     1755478601873 00:56:41  2.1h         10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5789046289 -                                 transport  1755478605993 00:56:45  2.1h         10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789046403 hPDJUkPUQNOLDRvLkMuDZQ:5789046289 direct     1755478606047 00:56:46  2.1h         10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5982567254 -                                 transport  1755478606908 00:56:46  2.1h         10.38.79.31  es-data2
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789047691 Zx2pLsOIRgOd08IBVlZS4g:5982567254 transport  1755478606935 00:56:46  2.1h         10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5982580000 -                                 transport  1755478616684 00:56:56  2.1h         10.38.79.31  es-data2
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789058574 Zx2pLsOIRgOd08IBVlZS4g:5982580000 transport  1755478616779 00:56:56  2.1h         10.38.49.21  es-data1
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5898069186 -                                 transport  1755478617098 00:56:57  2.1h         10.38.56.198 es-master
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5789058800 E_IDzeBLSBC0Ils0aJEFsA:5898069186 transport  1755478617341 00:56:57  2.1h         10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660467 -                                 transport  1755486405031 03:06:45  352.9ms      10.38.49.21  es-data1
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660468 hPDJUkPUQNOLDRvLkMuDZQ:5795660467 direct     1755486405031 03:06:45  352.7ms      10.38.49.21  es-data1
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660469 hPDJUkPUQNOLDRvLkMuDZQ:5795660467 direct     1755486405031 03:06:45  352.6ms      10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5989685651 -                                 transport  1755486405043 03:06:45  341.2ms      10.38.79.31  es-data2
indices:data/read/search[phase/query]    E_IDzeBLSBC0Ils0aJEFsA:5905281558 Zx2pLsOIRgOd08IBVlZS4g:5989685651 transport  1755486405044 03:06:45  340.1ms      10.38.56.198 es-master
indices:data/read/search[phase/query]    E_IDzeBLSBC0Ils0aJEFsA:5905281559 Zx2pLsOIRgOd08IBVlZS4g:5989685651 transport  1755486405044 03:06:45  340ms        10.38.56.198 es-master
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5989685675 -                                 transport  1755486405225 03:06:45  159ms        10.38.79.31  es-data2
indices:data/read/search[phase/fetch/id] E_IDzeBLSBC0Ils0aJEFsA:5905281842 Zx2pLsOIRgOd08IBVlZS4g:5989685675 transport  1755486405309 03:06:45  74.9ms       10.38.56.198 es-master
indices:data/read/search[phase/fetch/id] E_IDzeBLSBC0Ils0aJEFsA:5905281843 Zx2pLsOIRgOd08IBVlZS4g:5989685675 transport  1755486405309 03:06:45  74.8ms       10.38.56.198 es-master
indices:data/read/search[phase/fetch/id] Zx2pLsOIRgOd08IBVlZS4g:5989685749 Zx2pLsOIRgOd08IBVlZS4g:5989685675 direct     1755486405309 03:06:45  75.4ms       10.38.79.31  es-data2
indices:admin/refresh                    hPDJUkPUQNOLDRvLkMuDZQ:5795660518 -                                 transport  1755486405235 03:06:45  149.2ms      10.38.49.21  es-data1
indices:admin/refresh[s]                 hPDJUkPUQNOLDRvLkMuDZQ:5795660519 hPDJUkPUQNOLDRvLkMuDZQ:5795660518 transport  1755486405235 03:06:45  149.1ms      10.38.49.21  es-data1
indices:admin/refresh[s]                 Zx2pLsOIRgOd08IBVlZS4g:5989685688 hPDJUkPUQNOLDRvLkMuDZQ:5795660519 transport  1755486405235 03:06:45  148.9ms      10.38.79.31  es-data2
indices:admin/refresh[s][p]              Zx2pLsOIRgOd08IBVlZS4g:5989685689 Zx2pLsOIRgOd08IBVlZS4g:5989685688 direct     1755486405235 03:06:45  148.8ms      10.38.79.31  es-data2
indices:admin/refresh[s][r]              E_IDzeBLSBC0Ils0aJEFsA:5905281852 Zx2pLsOIRgOd08IBVlZS4g:5989685688 transport  1755486405324 03:06:45  59.9ms       10.38.56.198 es-master
cluster:monitor/nodes/stats              hPDJUkPUQNOLDRvLkMuDZQ:5795660552 -                                 transport  1755486405281 03:06:45  103.3ms      10.38.49.21  es-data1
cluster:monitor/nodes/stats[n]           hPDJUkPUQNOLDRvLkMuDZQ:5795660553 hPDJUkPUQNOLDRvLkMuDZQ:5795660552 direct     1755486405281 03:06:45  103.3ms      10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5989685721 -                                 transport  1755486405297 03:06:45  87.4ms       10.38.79.31  es-data2
indices:data/read/search[phase/query]    Zx2pLsOIRgOd08IBVlZS4g:5989685722 Zx2pLsOIRgOd08IBVlZS4g:5989685721 direct     1755486405297 03:06:45  87.2ms       10.38.79.31  es-data2
indices:data/read/search[phase/query]    E_IDzeBLSBC0Ils0aJEFsA:5905281780 Zx2pLsOIRgOd08IBVlZS4g:5989685721 transport  1755486405299 03:06:45  85.2ms       10.38.56.198 es-master
indices:data/read/search[phase/query]    E_IDzeBLSBC0Ils0aJEFsA:5905281785 Zx2pLsOIRgOd08IBVlZS4g:5989685721 transport  1755486405299 03:06:45  84.8ms       10.38.56.198 es-master
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281816 -                                 transport  1755486405303 03:06:45  80.5ms       10.38.56.198 es-master
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660582 -                                 transport  1755486405309 03:06:45  74.6ms       10.38.49.21  es-data1
indices:data/read/search[phase/fetch/id] hPDJUkPUQNOLDRvLkMuDZQ:5795660626 hPDJUkPUQNOLDRvLkMuDZQ:5795660582 direct     1755486405384 03:06:45  61.7micros   10.38.49.21  es-data1
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281853 -                                 transport  1755486405326 03:06:45  57.6ms       10.38.56.198 es-master
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660592 E_IDzeBLSBC0Ils0aJEFsA:5905281853 transport  1755486405327 03:06:45  57.4ms       10.38.49.21  es-data1
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660593 E_IDzeBLSBC0Ils0aJEFsA:5905281853 transport  1755486405327 03:06:45  57.4ms       10.38.49.21  es-data1
indices:data/read/search[phase/query]    Zx2pLsOIRgOd08IBVlZS4g:5989685755 E_IDzeBLSBC0Ils0aJEFsA:5905281853 transport  1755486405327 03:06:45  57.3ms       10.38.79.31  es-data2
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281869 -                                 transport  1755486405343 03:06:45  40.6ms       10.38.56.198 es-master
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660597 -                                 transport  1755486405343 03:06:45  40.7ms       10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660594 -                                 transport  1755486405343 03:06:45  40.9ms       10.38.49.21  es-data1
indices:admin/refresh                    hPDJUkPUQNOLDRvLkMuDZQ:5795660605 -                                 transport  1755486405344 03:06:45  40.2ms       10.38.49.21  es-data1
indices:admin/refresh[s]                 hPDJUkPUQNOLDRvLkMuDZQ:5795660606 hPDJUkPUQNOLDRvLkMuDZQ:5795660605 transport  1755486405344 03:06:45  40.1ms       10.38.49.21  es-data1
indices:admin/refresh[s]                 Zx2pLsOIRgOd08IBVlZS4g:5989685780 hPDJUkPUQNOLDRvLkMuDZQ:5795660606 transport  1755486405344 03:06:45  39.8ms       10.38.79.31  es-data2
indices:admin/refresh[s][p]              Zx2pLsOIRgOd08IBVlZS4g:5989685781 Zx2pLsOIRgOd08IBVlZS4g:5989685780 direct     1755486405344 03:06:45  39.6ms       10.38.79.31  es-data2
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660602 -                                 transport  1755486405344 03:06:45  40.4ms       10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660609 -                                 transport  1755486405346 03:06:45  38.4ms       10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660614 -                                 transport  1755486405350 03:06:45  33.6ms       10.38.49.21  es-data1
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660616 hPDJUkPUQNOLDRvLkMuDZQ:5795660614 direct     1755486405351 03:06:45  33.4ms       10.38.49.21  es-data1
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281881 -                                 transport  1755486405356 03:06:45  27.7ms       10.38.56.198 es-master
indices:data/read/msearch                E_IDzeBLSBC0Ils0aJEFsA:5905281895 E_IDzeBLSBC0Ils0aJEFsA:5905281881 direct     1755486405365 03:06:45  18.9ms       10.38.56.198 es-master
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281902 E_IDzeBLSBC0Ils0aJEFsA:5905281895 transport  1755486405365 03:06:45  18.4ms       10.38.56.198 es-master
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660622 E_IDzeBLSBC0Ils0aJEFsA:5905281902 transport  1755486405366 03:06:45  18.2ms       10.38.49.21  es-data1
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281896 E_IDzeBLSBC0Ils0aJEFsA:5905281895 transport  1755486405365 03:06:45  18.9ms       10.38.56.198 es-master
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660621 E_IDzeBLSBC0Ils0aJEFsA:5905281896 transport  1755486405365 03:06:45  18.6ms       10.38.49.21  es-data1
indices:data/read/search                 hPDJUkPUQNOLDRvLkMuDZQ:5795660617 -                                 transport  1755486405356 03:06:45  28.4ms       10.38.49.21  es-data1
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281885 -                                 transport  1755486405357 03:06:45  26.6ms       10.38.56.198 es-master
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660620 E_IDzeBLSBC0Ils0aJEFsA:5905281885 transport  1755486405358 03:06:45  26.4ms       10.38.49.21  es-data1
indices:data/read/search                 E_IDzeBLSBC0Ils0aJEFsA:5905281892 -                                 transport  1755486405364 03:06:45  19.4ms       10.38.56.198 es-master
indices:data/read/search[phase/query]    Zx2pLsOIRgOd08IBVlZS4g:5989685793 E_IDzeBLSBC0Ils0aJEFsA:5905281892 transport  1755486405365 03:06:45  18.9ms       10.38.79.31  es-data2
indices:data/read/search[phase/query]    E_IDzeBLSBC0Ils0aJEFsA:5905281893 E_IDzeBLSBC0Ils0aJEFsA:5905281892 direct     1755486405365 03:06:45  19.1ms       10.38.56.198 es-master
indices:data/read/search[phase/query]    E_IDzeBLSBC0Ils0aJEFsA:5905281894 E_IDzeBLSBC0Ils0aJEFsA:5905281892 direct     1755486405365 03:06:45  18.9ms       10.38.56.198 es-master
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5989685796 -                                 transport  1755486405377 03:06:45  7ms          10.38.79.31  es-data2
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660623 Zx2pLsOIRgOd08IBVlZS4g:5989685796 transport  1755486405377 03:06:45  6.4ms        10.38.49.21  es-data1
indices:data/read/search                 Zx2pLsOIRgOd08IBVlZS4g:5989685799 -                                 transport  1755486405383 03:06:45  806.6micros  10.38.79.31  es-data2
indices:data/read/search[phase/query]    Zx2pLsOIRgOd08IBVlZS4g:5989685800 Zx2pLsOIRgOd08IBVlZS4g:5989685799 direct     1755486405383 03:06:45  653.8micros  10.38.79.31  es-data2
indices:data/read/search[phase/query]    Zx2pLsOIRgOd08IBVlZS4g:5989685801 Zx2pLsOIRgOd08IBVlZS4g:5989685799 direct     1755486405384 03:06:45  580.2micros  10.38.79.31  es-data2
indices:data/read/search[phase/query]    hPDJUkPUQNOLDRvLkMuDZQ:5795660624 Zx2pLsOIRgOd08IBVlZS4g:5989685799 transport  1755486405384 03:06:45  191.2micros  10.38.49.21  es-data1
cluster:monitor/tasks/lists              E_IDzeBLSBC0Ils0aJEFsA:5905281908 -                                 transport  1755486405384 03:06:45  203.3micros  10.38.56.198 es-master
cluster:monitor/tasks/lists[n]           hPDJUkPUQNOLDRvLkMuDZQ:5795660625 E_IDzeBLSBC0Ils0aJEFsA:5905281908 transport  1755486405384 03:06:45  80.6micros   10.38.49.21  es-data1
cluster:monitor/tasks/lists[n]           Zx2pLsOIRgOd08IBVlZS4g:5989685802 E_IDzeBLSBC0Ils0aJEFsA:5905281908 transport  1755486405384 03:06:45  93.5micros   10.38.79.31  es-data2
cluster:monitor/tasks/lists[n]           E_IDzeBLSBC0Ils0aJEFsA:5905281909 E_IDzeBLSBC0Ils0aJEFsA:5905281908 direct     1755486405384 03:06:45  84.6micros   10.38.56.198 es-master

~~~

![1755486520243](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1755486520243.png)



![1755490368275](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1755490368275.png)



### synonym set Generally available; Added in 8.10.0

~~~
PUT _synonyms/my-synonyms-set

PUT _synonyms/my-synonyms-set/test-1
{
  "synonyms": "hello, hi, howdy"
}
GET _synonyms/my-synonyms-set
DELETE _synonyms/my-synonyms-set



~~~

### Stop words

~~~
https://www.elastic.co/guide/en/elasticsearch/reference/7.14/analysis-stop-tokenfilter.html#analysis-stop-tokenfilter-stop-words-by-lang

GET /_analyze
{
  "tokenizer": "standard",
  "filter": [
    {
      "type": "stop",
      "ignore_case": true,
      "stopwords": [ "_english_",  "Goes","near to" ,"بلة" ]
    }
  ],
  "text": "Li goes to park near to beijin القا بلة للتخصيص ومن"
}

_english_
_portuguese_
_french_
_german_
_spanish_
_indonesian_
_arabic_
~~~

~~~
synonym_graph：

PUT /test_index
{
  "settings": {
    "index": {
      "analysis": {
        "analyzer": {
          "my_synonym": {
            "tokenizer": "standard",
            "filter": [ "synonym_graph" ]
          }
        },
        "filter": {
          "my_stop": {
            "type": "stop",
            "stopwords": [ "bar" ]
          },
          "synonym_graph": {
            "type": "synonym_graph",
            "lenient": true,
            "expand":true,
            "synonyms": [ "foo, bar, baz" ]
          }
        }
      }
    }
  }
}
GET /test_index/_analyze
{
   "analyzer":"my_synonym",
  "text":"bar"
}

~~~

![1756706584436](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1756706584436.png)

NOTES： <font color=Red>"expand":false</font > ，only show as below:

~~~
{
  "tokens" : [
    {
      "token" : "foo",
      "start_offset" : 0,
      "end_offset" : 3,
      "type" : "SYNONYM",
      "position" : 0
    }
  ]
}
~~~



### ES indices yellow

~~~

GET /_cluster/health/

GET _cat/shards?h=index,shard,prirep,state,unassigned.reason&v


GET _cluster/allocation/explain?pretty

GET /_cluster/allocation/explain
{
"index": "spu_20251016084822",
"shard": 3,
"primary": true
}

~~~

## 2. DSL for Match

#### 2.1 Match + Function Script boot

~~~
{
  "from": 0,
  "size": 10,
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "must": [
            {
              "bool": {
                "should": [
                  {
                    "match": {
                      "productName": {
                        "query": "toy",
                        "operator": "OR",
                        "prefix_length": 0,
                        "max_expansions": 50,
                        "minimum_should_match": "2<75%",
                        "fuzzy_transpositions": true,
                        "lenient": false,
                        "zero_terms_query": "NONE",
                        "auto_generate_synonyms_phrase_query": true,
                        "boost": 1
                      }
                    }
                  }
                ],
                "adjust_pure_negative": true,
                "boost": 1
              }
            }
          ],
          "filter": [
            {
              "bool": {
                "must": [
                  {
                    "terms": {
                      "categoryId": [
                        22715
                      ],
                      "boost": 1
                    }
                  }
                ],
                "adjust_pure_negative": true,
                "boost": 1
              }
            }
          ],
          "adjust_pure_negative": true,
          "boost": 1
        }
      },
      "functions": [
        {
          "filter": {
            "match": {
              "categoryId": {
                "query": 22715,
                "operator": "OR",
                "prefix_length": 0,
                "max_expansions": 50,
                "fuzzy_transpositions": true,
                "lenient": false,
                "zero_terms_query": "NONE",
                "auto_generate_synonyms_phrase_query": true,
                "boost": 1
              }
            }
          },
          "weight": 100
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply",
      "max_boost": 3.4028235e+38,
      "boost": 1
    }
  },
  "explain": false,
  "_source": {
    "includes": [],
    "excludes": [
      "productDescription",
      "productOutline",
      "productGroups",
      "productImageUrl",
      "factories",
      "productGroups",
      "productKeyword1",
      "productKeyword2",
      "productKeyword3",
      "supplierState",
      "productInfoCopy",
      "categoryNameTreeCopy",
      "productKeyword",
      "productAttribute"
    ]
  },
  "track_total_hits": 2147483647
} 
~~~

#### 2.2 Match + constant score

~~~
{
  "from": 0,
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "bool": {
            "should": [
              {
                "match": {
                  "productName": {
                    "query": "toy",
                    "operator": "OR",
                    "prefix_length": 0,
                    "max_expansions": 50,
                    "minimum_should_match": "2<75%",
                    "fuzzy_transpositions": true,
                    "lenient": false,
                    "zero_terms_query": "NONE",
                    "auto_generate_synonyms_phrase_query": true,
                    "boost": 1
                  }
                }
              }
            ],
            "adjust_pure_negative": true,
            "boost": 1
          }
        }
      ],
      "should": [
        {
          "constant_score": {
            "filter": {
              "terms": {
                "categoryId": [
                  22715
                ],
                "boost": 1
              }
            },
            "boost": 100
          }
        }
      ],
      "adjust_pure_negative": true,
      "boost": 1
    }
  },
  "explain": false,
  "_source": {
    "includes": [],
    "excludes": [
      "productDescription",
      "productOutline",
      "productGroups",
      "productImageUrl",
      "factories",
      "productGroups",
      "productKeyword1",
      "productKeyword2",
      "productKeyword3",
      "supplierState",
      "productInfoCopy",
      "categoryNameTreeCopy",
      "productKeyword",
      "productAttribute"
    ]
  },
  "track_total_hits": 2147483647
} 

~~~

#### 2.3 Match with keyword + script_score

~~~
{
  "from": 0,
  "size": 10,
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "must": [
            {
              "bool": {
                "should": [
                  {
                    "match": {
                      "productName": {
                        "query": "toy",
                        "operator": "OR",
                        "prefix_length": 0,
                        "max_expansions": 50,
                        "minimum_should_match": "2<75%",
                        "fuzzy_transpositions": true,
                        "lenient": false,
                        "zero_terms_query": "NONE",
                        "auto_generate_synonyms_phrase_query": true,
                        "boost": 1
                      }
                    }
                  }
                ],
                "adjust_pure_negative": true,
                "boost": 1
              }
            }
          ],
          "adjust_pure_negative": true,
          "boost": 1
        }
      },
      "functions": [
 
        {
          "filter": {
            "match_all": {
              "boost": 1
            }
          },
          "script_score": {
            "script": {
              "source": "if(doc['productContentScore'].size() > 0 && doc['productContentScore'].value < 80){return 0.2 ;}else if(doc['productContentScore'].size() > 0  && doc['productContentScore'].value>=80 ){ return 0.4 ;} return 0.1;",
              "lang": "painless"
            }
          }
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply",
      "max_boost": 3.4028235e+38,
      "boost": 1
    }
  },
  "explain": true,
  "track_total_hits": 2147483647
} 
~~~

### <font color="red">New function_score + functions</font>

~~~
GET /spu/_search
{
  "size": 3, 
  "_source": ["productName","productBaseScore"], 
  "query": {
    "function_score": {
      "query": {
        "match": {
          "productName": "power bank"
        }
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "productBaseScore"
          }
        },
        {
          "filter": {
            "term": {"categoryId": 23683}
          },
          "weight": 0.25
        },
        { "weight": 1.0 }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  },
  "explain": true
}
~~~

![1778655155781](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1778655155781.png)

### 3 DSL for Agg

#### 3.1 Agg with Nested Category(L1~4)

~~~
{
  "from": 0,
  "size": 0,
  "query": {
    "bool": {
      "adjust_pure_negative": true,
      "boost": 1
    }
  },
  "explain": false,
  "_source": {
    "includes": [],
    "excludes": [
      "productDescription",
      "productOutline",
      "productGroups",
      "productImageUrl",
      "factories",
      "productGroups",
      "productKeyword1",
      "productKeyword2",
      "productKeyword3",
      "supplierState",
      "productInfoCopy",
      "categoryNameTreeCopy",
      "productKeyword",
      "productAttribute"
    ]
  },
  "track_total_hits": 2147483647,
  "aggregations": {
    "L1": {
      "nested": {
        "path": "categories"
      },
      "aggregations": {
        "L1": {
          "filter": {
            "bool": {
              "must": [
                {
                  "term": {
                    "categories.categoryLevel": {
                      "value": 1,
                      "boost": 1
                    }
                  }
                }
              ],
              "adjust_pure_negative": true,
              "boost": 1
            }
          },
          "aggregations": {
            "L1": {
              "terms": {
                "field": "categories.categoryId",
                "size": 5,
                "min_doc_count": 1,
                "shard_min_doc_count": 0,
                "show_term_doc_count_error": false,
                "order": [
                  {
                    "_count": "desc"
                  },
                  {
                    "_key": "asc"
                  }
                ]
              },
              "aggregations": {
                "reverse_categories": {
                  "reverse_nested": {},
                  "aggregations": {
                    "L2": {
                      "nested": {
                        "path": "categories"
                      },
                      "aggregations": {
                        "L2": {
                          "filter": {
                            "bool": {
                              "must": [
                                {
                                  "term": {
                                    "categories.categoryLevel": {
                                      "value": 2,
                                      "boost": 1
                                    }
                                  }
                                }
                              ],
                              "adjust_pure_negative": true,
                              "boost": 1
                            }
                          },
                          "aggregations": {
                            "L2": {
                              "terms": {
                                "field": "categories.categoryId",
                                "size": 5,
                                "min_doc_count": 1,
                                "shard_min_doc_count": 0,
                                "show_term_doc_count_error": false,
                                "order": [
                                  {
                                    "_count": "desc"
                                  },
                                  {
                                    "_key": "asc"
                                  }
                                ]
                              },
                              "aggregations": {
                                "reverse_categories": {
                                  "reverse_nested": {},
                                  "aggregations": {
                                    "L3": {
                                      "nested": {
                                        "path": "categories"
                                      },
                                      "aggregations": {
                                        "L3": {
                                          "filter": {
                                            "bool": {
                                              "must": [
                                                {
                                                  "term": {
                                                    "categories.categoryLevel": {
                                                      "value": 3,
                                                      "boost": 1
                                                    }
                                                  }
                                                }
                                              ],
                                              "adjust_pure_negative": true,
                                              "boost": 1
                                            }
                                          },
                                          "aggregations": {
                                            "L3": {
                                              "terms": {
                                                "field": "categories.categoryId",
                                                "size": 5,
                                                "min_doc_count": 1,
                                                "shard_min_doc_count": 0,
                                                "show_term_doc_count_error": false,
                                                "order": [
                                                  {
                                                    "_count": "desc"
                                                  },
                                                  {
                                                    "_key": "asc"
                                                  }
                                                ]
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
} 
~~~

#### 3.2 Agg with attr_name/value

~~~
{
  "from": 0,
  "size": 0,
  "query": {
    "bool": {
      "adjust_pure_negative": true,
      "boost": 1
    }
  },
  "explain": false,
  "_source": {
    "includes": [],
    "excludes": [
      "productDescription",
      "productOutline",
      "productGroups",
      "productImageUrl",
      "factories",
      "productGroups",
      "productKeyword1",
      "productKeyword2",
      "productKeyword3",
      "supplierState",
      "productInfoCopy",
      "categoryNameTreeCopy",
      "productKeyword",
      "productAttribute"
    ]
  },
  "track_total_hits": 2147483647,
  "aggregations": {
    "attrName": {
      "nested": {
        "path": "productAttribute"
      },
      "aggregations": {
        "attrName": {
          "terms": {
            "field": "productAttribute.attrName",
            "size": 5,
            "min_doc_count": 1,
            "shard_min_doc_count": 0,
            "show_term_doc_count_error": false,
            "order": [
              {
                "_count": "desc"
              },
              {
                "_key": "asc"
              }
            ]
          },
          "aggregations": {
            "reverse_productAttribute": {
              "reverse_nested": {},
              "aggregations": {
                "attrValue": {
                  "nested": {
                    "path": "productAttribute"
                  },
                  "aggregations": {
                    "attrValue": {
                      "terms": {
                        "field": "productAttribute.attrValue",
                        "size": 5,
                        "min_doc_count": 1,
                        "shard_min_doc_count": 0,
                        "show_term_doc_count_error": false,
                        "order": [
                          {
                            "_count": "desc"
                          },
                          {
                            "_key": "asc"
                          }
                        ]
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
} 
~~~



### 4. ISEARCH BIZ

	#### 4.1 PP/SP search api:

~~~
DESKTOP:
https://www.globalsources.com/api/agg-search/DESKTOP/v3/product/search
https://www.globalsources.com/api/agg-search/DESKTOP/v3/supplier/search

H5:
https://m.globalsources.com/api/agg-search/H5/v3/product/search
https://m.globalsources.com/api/agg-search/H5/v3/supplier/search


APP：
api/isearch-bff/v2/search-for-app
~~~



#### 4.2  ES tradeShows biz:

~~~
tradeShowStartDate/tradeShowEndDate：展会举行的开始/结束时间
tsStartDate/tsEndDate：展会前4后2 month的时间
~~~



### 5. ES vector search poc:

#### 5.1 ES mapping & setting & data init dense_vector

~~~
PUT passage_vectors2
{
    "mappings": {
        "properties": {
            "full_text": {
                "type": "text"
            },
            "creation_time": {
                "type": "date"
            },
            "paragraph": {
                "type": "nested",
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 2,
                        "index_options": {
                            "type": "hnsw"
                        }
                    },
                    "text": {
                        "type": "text",
                        "index": false
                    },
                    "language": {
                        "type": "keyword"
                    }
                }
            },
            "metadata": {
                "type": "nested",
                "properties": {
                    "key": {
                        "type": "keyword"
                    },
                    "value": {
                        "type": "text"
                    }
                }
            }
        }
    },
     "settings": {
    "number_of_replicas": 0
  }
}

POST passage_vectors2/_doc/1?refresh=true
{
  "full_text": "first paragraph another paragraph",
  "creation_time": "2019-05-04",
  "paragraph": [
    {
      "vector": [
        0.45,
        45
      ],
      "text": "first paragraph",
      "paragraph_id": "1",
      "language": "EN"
    },
    {
      "vector": [
        0.8,
        0.6
      ],
      "text": "another paragraph",
      "paragraph_id": "2",
      "language": "FR"
    }
  ],
  "metadata": [
    {
      "key": "author",
      "value": "Jane Doe"
    },
    {
      "key": "source",
      "value": "Internal Memo"
    }
  ]
}

POST passage_vectors2/_doc/2?refresh=true
{
  "full_text": "number one paragraph number two paragraph",
  "creation_time": "2020-05-04",
  "paragraph": [
    {
      "vector": [
        1.2,
        4.5
      ],
      "text": "number one paragraph",
      "paragraph_id": "1",
      "language": "EN"
    },
    {
      "vector": [
        -1,
        42
      ],
      "text": "number two paragraph",
      "paragraph_id": "2",
      "language": "EN"
    }
  ],
  "metadata": [
    {
      "key": "author",
      "value": "Jane Austen"
    },
    {
      "key": "source",
      "value": "Financial"
    }
  ]
}


---Filtering in nested KNN search
GET /passage_vectors2/_search
{
    "fields": ["full_text", "creation_time","paragraph.vector"],
    "_source": false,
    "knn": {
        "query_vector": [
            0.45,
            45
        ],
        "field": "paragraph.vector",
        "k": 2
    }
}
---Nested query in nested KNN search
POST passage_vectors2/_search
{
  "query" : {
    "nested" : {
      "path" : "paragraph",
        "query" : {
          "knn": {
            "query_vector": [
                0.45,
                45
            ],
            "field": "paragraph.vector",
            "num_candidates": 2
        }
      }
    }
  }
}

---Filtering on nested metadata
POST passage_vectors2/_search
{
    "fields": [
        "full_text"
    ],
    "_source": false,
    "knn": {
        "query_vector": [0.45,45],
        "field": "paragraph.vector",
        "k": 2,
        "filter": [
            {"match": {"paragraph.language": "EN"}},
            {"range": { "creation_time": { "gte": "2019-05-01", "lte": "2019-05-05"}}}
        ]
    }
}

---Filtering by sibling nested fields in nested KNN search
POST passage_vectors2/_search
{
    "fields": [
        "full_text","metadata.*"
    ],
    "_source": false,
    "knn": {
        "query_vector": [0.45, 45],
        "field": "paragraph.vector",
        "k": 2,
        "filter": {
            "nested": {
                "path": "metadata",
                "query": {
                    "bool": {
                        "must": [
                            { "match": { "metadata.key": "author" } },
                            { "match": { "metadata.value": "Austen" } }
                        ]
                    }
                }
            }
        }
    }
}


---Nested kNN Search with Inner hits
POST passage_vectors2/_search
{
    "fields": [
        "creation_time",
        "full_text"
    ],
    "_source": false,
    "knn": {
        "query_vector": [
            0.45,
            45
        ],
        "field": "paragraph.vector",
        "k": 2,
        "num_candidates": 2,
        "inner_hits": {
            "_source": false,
            "fields": [
                "paragraph.text"
            ],
            "size": 1
        }
    }
}

--Hybrid search with  KNN + match nested fields
POST passage_vectors2/_search
{
  "size": 3,
  "query": {
    "bool": {
      "should": [
        {
          "nested": {
            "path": "metadata",
            "query": {
              "match": {
                "metadata.value": {
                  "query": "Austen",
                  "boost": 1
                }
              }
            }
          }
        },
        {
          "nested": {
            "path": "paragraph",
            "query": {
              "knn": {
                "query_vector": [
                  0.45,
                  45
                ],
                "field": "paragraph.vector",
                "num_candidates": 2
              }
            }
          }
        }
      ]
    }
  }
}
~~~

#### 5.2 ES mapping & setting & data init for sparse_vector（score:dot product)

~~~
POST marketplace/_doc/1
{
  "title": "playstation 5 - special offer",
  "query_boost": [
    {"playstation": 3, "game console": 1}
  ]
}

POST marketplace/_doc/2
{
  "title": "playstation controller"
}

POST marketplace/_doc/3
{
  "title": "High fructose snack bar with artificial flavor"
}
 
POST marketplace/_doc/4
{
  "title": "Snack bar with whole food ingredients",
  "customer_types": {
    "healthy-conscious": 3
  }
} 



GET marketplace/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "title": "playstation"
          }
        }
      ]
    }
  }

}

GET marketplace/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "title": "playstation"
          }
        }
      ],
      "should": [
        {
          "sparse_vector": {
            "field": "query_boost",
            "query_vector": {
              "playstation": 1
            }
          }
        }
      ]
    }
  }
}
 
~~~

### 5.3 Semantic search 

##### 5.3.1  Semantic search with `semantic_text`

~~~
				
PUT semantic-embeddings					
{
  "mappings": {
    "properties": {
      "content": {
        "type": "semantic_text"
      }
    }
  }
}

				
PUT semantic-embeddings-custom
{
  "mappings": {
    "properties": {
      "content": {
        "type": "semantic_text",
        "inference_id": ".multilingual-e5-small-elasticsearch",
        "index_options": {
          "dense_vector": {
            "type": "bbq_hnsw",
            "m": 32,
            "ef_construction": 200
          }
        }
      }
    }
  }
}
--m:The number of neighbors each node will be connected to in the HNSW graph. Higher values improve recall but increase memory usage. Default is 16.
--ef_construction:Number of candidates considered during graph construction. Higher values improve index quality but slow down indexing. Default is 100.

POST _reindex?wait_for_completion=false
{
  "source": {
    "index": "test-data",
    "size": 10
  },
  "dest": {
    "index": "semantic-embeddings"
  }
}

				
GET _tasks/<task_id>
POST _tasks/<task_id>/_cancel

GET semantic-embeddings/_search
{
  "query": {
    "match": {
      "content": {
        "query": "What causes muscle soreness after running?"
      }
    }
  }
}		
		
~~~

![1772761063751](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1772761063751.png)



##### 5.3.2 Semantic search with the inference API

~~~
				
PUT _inference/sparse_embedding/elser_embeddings				
{
  "service": "elasticsearch",
  "service_settings": {
    "num_allocations": 1,
    "num_threads": 1,
    "model_id": ".elser_model_2_linux-x86_64"
  }
}
				
PUT elser-embeddings					
{
  "mappings": {
    "properties": {
      "content_embedding": {
        "type": "sparse_vector"
      },
      "content": {
        "type": "text"
      }
    }
  }
}
			
PUT _ingest/pipeline/elser_embeddings_pipeline					
{
  "processors": [
    {
      "inference": {
        "model_id": "elser_embeddings",
        "input_output": {
          "input_field": "content",
          "output_field": "content_embedding"
        }
      }
    }
  ]
}

POST _reindex?wait_for_completion=false
{
  "source": {
    "index": "test-data",
    "size": 50
  },
  "dest": {
    "index": "elser-embeddings",
    "pipeline": "elser_embeddings_pipeline"
  }
}		

GET elser-embeddings/_search
{
  "query":{
    "sparse_vector":{
      "field": "content_embedding",
      "inference_id": ".elser-2-elasticsearch",
      "query": "How to avoid muscle soreness after running?"
    }
  },
  "_source": [
    "id",
    "content"
  ]
}
		
~~~

##### 5.3.3 Semantic search with ELSER

~~~
PUT my-index
{
  "mappings": {
  	"_source": {
      "excludes": [
        "content_embedding"
      ]
    },
    "properties": {
      "content_embedding": {
        "type": "sparse_vector"
      },
      "content": {
        "type": "text"
      }
    }
  }
}

PUT _ingest/pipeline/elser-v2-test
{
  "processors": [
    {
      "inference": {
        "model_id": ".elser_model_2_linux-x86_64",
        "input_output": [
          {
            "input_field": "content",
            "output_field": "content_embedding"
          }
        ]
      }
    }
  ]
}

POST _reindex?wait_for_completion=false
{
  "source": {
    "index": "test-data",
    "size": 50
  },
  "dest": {
    "index": "my-index",
    "pipeline": "elser-v2-test"
  }
}

GET my-index/_search
{
   "query":{
      "sparse_vector":{
         "field": "content_embedding",
         "inference_id": ".elser-2-elasticsearch",
         "query": "How to avoid muscle soreness after running?"
      }
   }
}
~~~

##### 5.3.4 Hybrid search with `semantic_text`

~~~
PUT semantic-embeddings2
{
  "mappings": {
    "properties": {
      "semantic_text": {
        "type": "semantic_text"
      },
      "content": {
        "type": "text",
        "copy_to": "semantic_text"
      }
    }
  }
}

POST _reindex?wait_for_completion=false
{
  "source": {
    "index": "test-data",
    "size": 10
  },
  "dest": {
    "index": "semantic-embeddings2"
  }
}

get /_tasks/xKPVhjIxQmaRaJgGfU4eQw:10182723
POST /_tasks/xKPVhjIxQmaRaJgGfU4eQw:10182723/_cancel

GET semantic-embeddings2/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "match": {
                "content": "How to avoid muscle soreness while running?"
              }
            }
          }
        },
        {
          "standard": {
            "query": {
              "semantic": {
                "field": "semantic_text",
                "query": "How to avoid muscle soreness while running?"
              }
            }
          }
        }
      ]
    }
  }
}

~~~



#### 5.4 Detail <span style="color:red">tuning</span>

~~~~
flat --> int8_flat --> int4_flat --> bbq_flat --> hnsw --> int8_hnsw --> int4_hnsw --> bbq_hnsw

				
PUT my-index-000001					
{
    "mappings": {
        "properties": {
            "text_embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index_options": {
                    "type": "bbq_hnsw"
                }
            }
        }
    }
}

		
~~~~

~~~

~~~

![1776410046086](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1776410046086.png)









~~~
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

conda create -n conda_env_13 
conda activate conda_env_13
conda install python=3.13.13
# 激活环境后，在当前环境安装
conda install numpy
# 或者，不激活环境，直接用 -n 指定环境名安装
conda install -n myenv numpy
~~~

