from util.PostgreDBUtil import PostgreSQLUtil
import pandas as pd

if __name__ == '__main__':
    base_sql = "SELECT product_id ,product_name ,category_id ,org_id  FROM product_grp.online_product op where target_website_status ='Online' and website_type ='GSOL'"
    sql = base_sql + "and org_id = %s"
    params = (2008852566804,)


    with PostgreSQLUtil() as db:
        # 示例1：普通查询
        total_count = db.execute_query_count(sql,params)
        for i in range(2):
            paginated_result = db.query_paginated(
                sql_query=sql,
                page_no=i+1,
                page_size=10,
                params=params,
                total_count=total_count,
                order_by="product_id DESC"
            )
            print("******",i)
            df = pd.DataFrame(paginated_result.data, columns=paginated_result.columns)
            print(df)