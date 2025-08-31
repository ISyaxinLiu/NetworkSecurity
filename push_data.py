import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

# PostgreSQL连接配置 - 使用你的ns数据库
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ns")  # 使用你创建的ns数据库
DB_USER = os.getenv("DB_USER", "nsproject")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123")

print(f"PostgreSQL连接信息: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def csv_to_json_convertor(self, file_path):
        """
        CSV转JSON - 保持原有逻辑不变
        """
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def insert_data_postgresql(self, records, database, table_name):
        """
        插入数据到PostgreSQL - 替代insert_data_mongodb
        
        Args:
            records: 要插入的记录列表
            database: 数据库名称 (这里会是"ns")
            table_name: 表名称 (对应MongoDB的collection)
        """
        try:
            self.database = database
            self.table_name = table_name
            self.records = records
            
            # 创建PostgreSQL连接
            self.connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=database,  # 使用ns数据库
                user=DB_USER,
                password=DB_PASSWORD
            )
            
            print(f"✅ 成功连接到数据库: {database}")
            
            # 创建游标
            cursor = self.connection.cursor()
            
            # 创建表（如果不存在）
            self._create_table_if_not_exists(cursor, table_name)
            
            # 批量插入数据
            success_count = self._insert_records_batch(cursor, records, table_name)
            
            # 提交事务
            self.connection.commit()
            print(f"✅ 数据提交成功")
            
            # 关闭连接
            cursor.close()
            self.connection.close()
            
            return success_count
            
        except Exception as e:
            print(f"❌ 数据库操作失败: {str(e)}")
            if hasattr(self, 'connection'):
                self.connection.rollback()
                self.connection.close()
            raise NetworkSecurityException(e, sys)
    
    def _create_table_if_not_exists(self, cursor, table_name):
        """
        创建表结构
        """
        try:
            # 根据你的CSV文件字段创建表
            create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id SERIAL PRIMARY KEY,
                "having_IP_Address" INTEGER,
                "URL_Length" INTEGER,
                "Shortining_Service" INTEGER,
                "having_At_Symbol" INTEGER,
                "double_slash_redirecting" INTEGER,
                "Prefix_Suffix" INTEGER,
                "having_Sub_Domain" INTEGER,
                "SSLfinal_State" INTEGER,
                "Domain_registeration_length" INTEGER,
                "Favicon" INTEGER,
                "port" INTEGER,
                "HTTPS_token" INTEGER,
                "Request_URL" INTEGER,
                "URL_of_Anchor" INTEGER,
                "Links_in_tags" INTEGER,
                "SFH" INTEGER,
                "Submitting_to_email" INTEGER,
                "Abnormal_URL" INTEGER,
                "Redirect" INTEGER,
                "on_mouseover" INTEGER,
                "RightClick" INTEGER,
                "popUpWidnow" INTEGER,
                "Iframe" INTEGER,
                "age_of_domain" INTEGER,
                "DNSRecord" INTEGER,
                "web_traffic" INTEGER,
                "Page_Rank" INTEGER,
                "Google_Index" INTEGER,
                "Links_pointing_to_page" INTEGER,
                "Statistical_report" INTEGER,
                "Result" INTEGER,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            '''
            
            cursor.execute(create_table_sql)
            print(f"✅ 表 {table_name} 创建成功或已存在")
            
        except Exception as e:
            print(f"❌ 创建表失败: {str(e)}")
            raise e
    
    def _insert_records_batch(self, cursor, records, table_name):
        """
        批量插入记录
        """
        try:
            if not records:
                return 0
            
            print(f"开始插入 {len(records)} 条记录到表 {table_name}")
            
            # 获取字段名
            first_record = records[0]
            columns = [f'"{col}"' for col in first_record.keys()]  # 加双引号处理特殊字符
            
            # 准备插入SQL
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            insert_sql = f'''
            INSERT INTO "{table_name}" ({columns_str}) 
            VALUES ({placeholders})
            '''
            
            # 准备数据
            data_tuples = []
            for record in records:
                values = tuple(record[col.strip('"')] for col in [col.strip('"') for col in columns])
                data_tuples.append(values)
            
            # 批量插入 - 分批处理避免内存问题
            batch_size = 1000
            inserted_count = 0
            
            for i in range(0, len(data_tuples), batch_size):
                batch = data_tuples[i:i + batch_size]
                cursor.executemany(insert_sql, batch)
                inserted_count += len(batch)
                print(f"已插入 {inserted_count}/{len(records)} 条记录")
            
            print(f"✅ 成功插入 {inserted_count} 条记录")
            return inserted_count
            
        except Exception as e:
            print(f"❌ 批量插入失败: {str(e)}")
            raise e

if __name__ == '__main__':
    # 使用你的ns数据库
    FILE_PATH = "Network_Data/phisingData.csv"  
    DATABASE = "ns"  # 使用你创建的ns数据库
    Collection = "NetworkData"  # 这将成为PostgreSQL中的表名
    
    try:
        print("🚀 开始ETL处理...")
        
        networkobj = NetworkDataExtract()
        
        print("📖 读取和转换CSV文件...")
        records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
        print(f"✅ 转换完成，共 {len(records)} 条记录")
        
        # 显示前3条记录作为样本
        print("\n📋 数据样本（前3条记录）:")
        for i, record in enumerate(records[:3]):
            print(f"记录 {i+1}: {dict(list(record.items())[:5])}...")  # 只显示前5个字段避免太长
        
        print(f"\n💾 开始插入数据到PostgreSQL数据库 '{DATABASE}'...")
        no_of_records = networkobj.insert_data_postgresql(records, DATABASE, Collection)
        
        print(f"\n🎉 ETL流程完成！")
        print(f"📊 统计信息:")
        print(f"   - 处理文件: {FILE_PATH}")
        print(f"   - 目标数据库: {DATABASE}")
        print(f"   - 目标表: {Collection}")
        print(f"   - 成功插入: {no_of_records} 条记录")
        
    except Exception as e:
        print(f"❌ ETL流程失败: {str(e)}")
        sys.exit(1)