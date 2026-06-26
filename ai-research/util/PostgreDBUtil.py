import psycopg2
from psycopg2 import sql, extras
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import os
import pandas as pd
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PageResult:
    """分页结果封装类"""
    data: List[Dict[str, Any]]  # 当前页数据
    total: int  # 总记录数
    page: int  # 当前页码
    page_size: int  # 每页大小
    total_pages: int  # 总页数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'data': self.data,
            'pagination': {
                'total': self.total,
                'page': self.page,
                'page_size': self.page_size,
                'total_pages': self.total_pages,
                'has_next': self.page < self.total_pages,
                'has_prev': self.page > 1
            }
        }


class PostgreSQLUtil:
    """PostgreSQL数据库工具类"""
    
    def __init__(self):
        """
        初始化数据库连接配置
        
        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名
            user: 用户名
            password: 密码
        """
        self.connection_params = {
            'host': os.getenv("PG_DB_URL"),
            'port': os.getenv("PG_DB_PORT"),
            'database': os.getenv("PG_DB_NAME"),
            'user': os.getenv("PG_DB_USER"),
            'password': os.getenv("PG_DB_PASSWORD")
        }


        self.connection = None
        self.cursor = None
    
    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.cursor = self.connection.cursor(cursor_factory=extras.RealDictCursor)
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def disconnect(self):
        """关闭数据库连接"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def __enter__(self):
        """支持上下文管理器"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时关闭连接"""
        self.disconnect()
    
    def execute_query(self, sql_query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        执行查询SQL并返回结果
        
        Args:
            sql_query: SQL查询语句
            params: SQL参数（元组形式）
            
        Returns:
            查询结果列表，每行数据为字典格式
        """
        try:
            if params:
                self.cursor.execute(sql_query, params)
            else:
                self.cursor.execute(sql_query)
            
            results = self.cursor.fetchall()
            # 将RealDictRow转换为普通字典列表
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            raise
    
    def execute_update(self, sql_query: str, params: Optional[Tuple] = None) -> int:
        """
        执行更新SQL（INSERT, UPDATE, DELETE）
        
        Args:
            sql_query: SQL语句
            params: SQL参数（元组形式）
            
        Returns:
            受影响的行数
        """
        try:
            if params:
                self.cursor.execute(sql_query, params)
            else:
                self.cursor.execute(sql_query)
            
            self.connection.commit()
            affected_rows = self.cursor.rowcount
            logger.info(f"更新成功，影响行数: {affected_rows}")
            return affected_rows
        except Exception as e:
            self.connection.rollback()
            logger.error(f"更新执行失败: {e}")
            raise
    
    def query_paginated(self, 
                        sql_query: str, 
                        page: int = 1, 
                        page_size: int = 10,
                        params: Optional[Tuple] = None,
                        order_by: Optional[str] = None) -> PageResult:
        """
        执行分页查询
        
        Args:
            sql_query: 原始SQL查询语句（不含ORDER BY和LIMIT）
            page: 页码，从1开始
            page_size: 每页记录数
            params: SQL参数（元组形式）
            order_by: 排序子句，例如 "id DESC" 或 "created_at ASC"
            
        Returns:
            PageResult对象，包含分页数据和分页信息
        """
        # 参数验证
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 10
        
        # 构建计数查询
        count_sql = f"SELECT COUNT(*) AS total FROM ({sql_query}) AS subquery"
        
        try:
            # 1. 获取总记录数
            self.cursor.execute(count_sql, params)
            total_result = self.cursor.fetchone()
            total = total_result['total'] if total_result else 0
            
            # 2. 构建分页查询
            offset = (page - 1) * page_size
            
            # 构建完整的查询SQL
            paginated_sql = sql_query
            
            # 添加ORDER BY（如果提供）
            if order_by:
                paginated_sql = f"{paginated_sql} ORDER BY {order_by}"
            
            # 添加LIMIT和OFFSET
            paginated_sql = f"{paginated_sql} LIMIT %s OFFSET %s"
            
            # 合并参数
            if params:
                paginated_params = params + (page_size, offset)
            else:
                paginated_params = (page_size, offset)
            
            # 3. 执行分页查询
            self.cursor.execute(paginated_sql, paginated_params)
            results = self.cursor.fetchall()
            data = [dict(row) for row in results]
            
            # 4. 计算总页数
            total_pages = (total + page_size - 1) // page_size if total > 0 else 0
            
            return PageResult(
                data=data,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages
            )
            
        except Exception as e:
            logger.error(f"分页查询失败: {e}")
            raise
    
    def query_paginated_with_sql(self, 
                                 sql_query: str, 
                                 page: int = 1, 
                                 page_size: int = 10,
                                 params: Optional[Tuple] = None) -> PageResult:
        """
        支持完整SQL的分页查询（包含ORDER BY）
        注意：此方法要求SQL中不包含LIMIT和OFFSET，会自动添加
        
        Args:
            sql_query: 完整的SQL查询语句（不含LIMIT和OFFSET）
            page: 页码，从1开始
            page_size: 每页记录数
            params: SQL参数（元组形式）
            
        Returns:
            PageResult对象
        """
        return self.query_paginated(sql_query, page, page_size, params)
    
    def execute_batch_insert(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        """
        批量插入数据
        
        Args:
            table_name: 表名
            data_list: 要插入的数据列表，每个元素为字典格式
            
        Returns:
            插入的行数
        """
        if not data_list:
            return 0
        
        try:
            columns = list(data_list[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            sql_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # 准备批量数据
            values_list = [[row[col] for col in columns] for row in data_list]
            
            extras.execute_values(
                self.cursor, 
                sql_query, 
                values_list,
                page_size=100
            )
            
            self.connection.commit()
            logger.info(f"批量插入成功，插入行数: {len(data_list)}")
            return len(data_list)
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"批量插入失败: {e}")
            raise


# 使用示例
def example_usage():
    """使用示例"""
    # 配置数据库连接
    # db_config = {
    #     'host': os.getenv("PG_DB_URL"),
    #     'port': os.getenv("PG_DB_PORT"),
    #     'database': os.getenv("PG_DB_NAME"),
    #     'user': os.getenv("PG_DB_USER"),
    #     'password': os.getenv("PG_DB_PASSWORD")
    # }
    #
    # 方式1：使用上下文管理器
    with PostgreSQLUtil() as db:
        # 示例1：普通查询
        print("=== 普通查询示例 ===")
        base_sql = "SELECT product_id ,product_name ,category_id ,org_id  FROM product_grp.online_product op where target_website_status ='Online' and website_type ='GSOL'"
        sql =  base_sql + "and product_id= %s"
        results = db.execute_query(sql, (1201198991,))
        for row in results:
            print(row)
        
        # 示例2：分页查询
        print("\n=== 分页查询示例 ===")
        paginated_result = db.query_paginated(
            sql_query=base_sql + " and org_id = %s",
            page=2,
            page_size=10,
            params=(2008852566804,),
            order_by="product_id DESC"
        )
        print(f"总记录数: {paginated_result.total}")
        print(f"总页数: {paginated_result.total_pages}")
        print(f"当前页数据: {paginated_result.data}")
        print(f"分页信息字典: {paginated_result.to_dict()}")
        
        # 示例3：使用完整SQL的分页
        # print("\n=== 完整SQL分页示例 ===")
        # result2 = db.query_paginated_with_sql(
        #     sql_query="SELECT id, name, age FROM users WHERE age > 18 ORDER BY id",
        #     page=1,
        #     page_size=5
        # )
        # print(f"数据: {result2.data}")

    
    # 方式2：手动管理连接
    # print("\n=== 手动管理连接 ===")
    # db = PostgreSQLUtil(**db_config)
    # try:
    #     db.connect()
    #     results = db.execute_query("SELECT * FROM users LIMIT 5")
    #     print(results)
    # finally:
    #     db.disconnect()


if __name__ == "__main__":
    example_usage()