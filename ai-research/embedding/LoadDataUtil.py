from util.PostgreDBUtil import PostgreSQLUtil


if __name__ == '__main__':
    base_sql = "SELECT product_id ,product_name ,category_id ,org_id  FROM product_grp.online_product op where target_website_status ='Online' and website_type ='GSOL'"
    sql = base_sql + "and product_id= %s"


    with PostgreSQLUtil() as db:
        # 示例1：普通查询

        # for i in
        paginated_result = db.query_paginated(
            sql_query=base_sql + " and org_id = %s",
            page_no=2,
            page_size=10,
            params=(2008852566804,),
            order_by="product_id DESC"
        )