from util.PostgreDBUtil import PostgreSQLUtil
import pandas as pd
from util.ExcelExporter import ExcelExporter
if __name__ == '__main__':
    base_sql = "SELECT product_id ,product_name ,category_id ,org_id  FROM product_grp.online_product op where target_website_status ='Online' and website_type ='GSOL'"
    sql = base_sql + "and org_id = %s"
    params = (2008852566804,)

    sql_cat = "select category_id  from product_grp.product_category pc where delete_flag =false  and category_level =4 order by create_date asc"
    sql_product = base_sql + "and category_id = %s order by create_date asc limit 100"
    with PostgreSQLUtil() as db:
        # 示例1：普通查询
        cat_result = db.execute_query(sql_query=sql_cat)
        cat_list =  [row['category_id'] for row in cat_result]
        print(f"Total cat size:{len(cat_list)}")
        all_data = []
        for idx, cat_id in enumerate(cat_list):
            pp_result = db.execute_query(sql_query=sql_product,params=(cat_id,))
            all_data.extend(pp_result)
            print(f"Batch No.{idx} for Cat id:{cat_id} get PP size:{len(pp_result)}")
            # if idx == 10:
            #     break
        print(f"Total PP count:{len(all_data)}")

        ExcelExporter().export_list(data=all_data,filename="../data/export/products.xlsx",sheet_name="PP")
        # total_count = db.execute_query_count(sql,params)
        # for i in range(2):
        #     paginated_result = db.query_paginated(
        #         sql_query=sql,
        #         page_no=i+1,
        #         page_size=10,
        #         params=params,
        #         total_count=total_count,
        #         order_by="product_id ASC"
        #     )
        #     print("******",i)
        #     df = pd.DataFrame(paginated_result.data, columns=paginated_result.columns)
        #     print(df)