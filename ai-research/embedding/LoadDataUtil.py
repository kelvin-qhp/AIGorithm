from util.PostgreDBUtil import PostgreSQLUtil
import pandas as pd
from util.ExcelExporter import ExcelExporter


def exportProduct():
    base_sql = "SELECT product_id ,product_name ,category_id ,org_id  FROM product_grp.online_product op where target_website_status ='Online' and website_type ='GSOL'"
    sql_cat = "select category_id  from product_grp.product_category pc where delete_flag =false  and category_level =4 order by create_date asc"
    sql_product = base_sql + "and category_id = %s order by create_date asc limit 100"

    with PostgreSQLUtil() as db:
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
        print(f"success to export product size:{len(all_data)}")


def exportCategory():
    sql_cat_full_path = """
       SELECT \
           l1.category_id AS l1_category_id,\
           l1.desc_en AS l1_category_name,\
           l2.category_id AS l2_category_id,\
           l2.desc_en AS l2_category_name,\
           l3.category_id AS l3_category_id,\
           l3.desc_en AS l3_category_name,\
           l4.category_id AS l4_category_id,\
           l4.desc_en AS l4_category_name\
       FROM product_grp.product_category l1\
       LEFT JOIN product_grp.product_category l2 ON l2.parent_category_id = l1.category_id AND l2.category_level = 2 and l2.delete_flag=false \
       LEFT JOIN product_grp.product_category l3 ON l3.parent_category_id = l2.category_id AND l3.category_level = 3 and l3.delete_flag=false \
       LEFT JOIN product_grp.product_category l4 ON l4.parent_category_id = l3.category_id AND l4.category_level = 4 and l4.delete_flag=false \
       WHERE l1.category_level = 1 AND l1.delete_flag=false\
       ORDER BY l1.category_id, l2.category_id, l3.category_id, l4.category_id
       """
    with PostgreSQLUtil() as db:
        data, cols = db.execute_query_with_columns(sql_query=sql_cat_full_path)

    ExcelExporter().export_list(data=data, filename="../data/export/category.xlsx", sheet_name="Category")
    print(f"success to export full path cat columns:{cols} for data size:{len(data)}")

if __name__ == '__main__':
    # exportProduct()

    exportCategory()


