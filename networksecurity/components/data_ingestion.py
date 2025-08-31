from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

## configuration of the Data Ingestion Config
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd

# PostgreSQL相关导入 - 替代pymongo
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL连接配置 - 替代MONGO_DB_URL
POSTGRES_HOST = os.getenv("DB_HOST", "localhost")
POSTGRES_PORT = os.getenv("DB_PORT", "5432")
POSTGRES_USER = os.getenv("DB_USER", "nsproject")
POSTGRES_PASSWORD = os.getenv("DB_PASSWORD", "123")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        """
        从PostgreSQL数据库读取数据 - 替代原来的MongoDB读取方法
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name  # 这里实际上是表名
            
            logging.info(f"开始从PostgreSQL读取数据: 数据库={database_name}, 表={collection_name}")
            
            # 构建PostgreSQL连接字符串
            postgres_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{database_name}"
            
            # 方法1: 使用SQLAlchemy (推荐，更简单)
            engine = create_engine(postgres_url)
            
            # 读取数据，注意表名用双引号包围以处理特殊字符
            query = f'SELECT * FROM "{collection_name}"'
            df = pd.DataFrame()
            
            try:
                df = pd.read_sql(query, engine)
                logging.info(f"使用SQLAlchemy成功读取数据: {len(df)} 行, {len(df.columns)} 列")
            except Exception as sqlalchemy_error:
                logging.warning(f"SQLAlchemy读取失败，尝试使用psycopg2: {sqlalchemy_error}")
                
                # 方法2: 备用方案 - 使用psycopg2
                connection = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    database=database_name,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD
                )
                
                cursor = connection.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query)
                rows = cursor.fetchall()
                df = pd.DataFrame([dict(row) for row in rows])
                
                cursor.close()
                connection.close()
                logging.info(f"使用psycopg2成功读取数据: {len(df)} 行")
            
            finally:
                if 'engine' in locals():
                    engine.dispose()
            
            # 数据清理 - 移除数据库自动生成的列
            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)
                logging.info("移除了数据库ID列")
            
            if "created_at" in df.columns.to_list():
                df = df.drop(columns=["created_at"], axis=1)
                logging.info("移除了创建时间列")
            
            # 处理缺失值 - 与原代码保持一致
            df.replace({"na": np.nan}, inplace=True)
            
            logging.info(f"数据读取完成，最终数据形状: {df.shape}")
            logging.info(f"列名: {df.columns.tolist()}")
            
            # 检查必要的列是否存在
            if "Result" not in df.columns:
                raise ValueError("目标列 'Result' 不存在于数据中")
            
            return df
            
        except Exception as e:
            logging.error(f"从PostgreSQL读取数据失败: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        将数据导出到特征存储 - 与原代码完全相同
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            
            # 创建文件夹
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # 保存为CSV文件
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            
            logging.info(f"数据成功导出到特征存储: {feature_store_file_path}")
            logging.info(f"导出数据形状: {dataframe.shape}")
            
            return dataframe
            
        except Exception as e:
            logging.error(f"导出数据到特征存储失败: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        将数据分割为训练集和测试集 - 与原代码完全相同
        """
        try:
            # 进行训练测试分割，添加分层采样确保数据分布一致
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,  # 添加随机种子确保可重复性
                stratify=dataframe['Result'] if 'Result' in dataframe.columns else None
            )
            
            logging.info("Performed train test split on the dataframe")
            logging.info(f"训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            # 创建目录
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            # 保存训练集
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            # 保存测试集
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            
            logging.info(f"Exported train and test file path.")
            logging.info(f"训练文件: {self.data_ingestion_config.training_file_path}")
            logging.info(f"测试文件: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            logging.error(f"数据分割失败: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        """
        启动数据摄取流程 - 与原代码逻辑完全一致，只是数据源改为PostgreSQL
        """
        try:
            logging.info("开始数据摄取流程")
            
            # 步骤1: 从PostgreSQL读取数据 (替代从MongoDB读取)
            logging.info("从PostgreSQL数据库读取数据")
            dataframe = self.export_collection_as_dataframe()
            
            # 数据质量检查
            logging.info(f"读取数据概览:")
            logging.info(f"  - 数据形状: {dataframe.shape}")
            logging.info(f"  - 缺失值数量: {dataframe.isnull().sum().sum()}")
            if 'Result' in dataframe.columns:
                result_counts = dataframe['Result'].value_counts()
                logging.info(f"  - 目标变量分布: {result_counts.to_dict()}")
            
            # 步骤2: 导出到特征存储
            logging.info("导出数据到特征存储")
            dataframe = self.export_data_into_feature_store(dataframe)
            
            # 步骤3: 分割训练测试数据
            logging.info("分割数据为训练集和测试集")
            self.split_data_as_train_test(dataframe)
            
            # 创建数据摄取工件 - 与原代码完全相同
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info("数据摄取流程完成")
            logging.info(f"训练文件: {dataingestionartifact.trained_file_path}")
            logging.info(f"测试文件: {dataingestionartifact.test_file_path}")
            
            return dataingestionartifact

        except Exception as e:
            logging.error(f"数据摄取流程失败: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def get_database_info(self):
        """
        获取数据库信息的辅助方法 - 新增功能，用于调试
        """
        try:
            database_name = self.data_ingestion_config.database_name
            table_name = self.data_ingestion_config.collection_name
            
            postgres_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{database_name}"
            engine = create_engine(postgres_url)
            
            # 获取表信息
            info_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(CASE WHEN "Result" = 1 THEN 1 END) as phishing_count,
                COUNT(CASE WHEN "Result" = -1 THEN 1 END) as legitimate_count
            FROM "{table_name}"
            """
            
            result = pd.read_sql(info_query, engine)
            engine.dispose()
            
            info = {
                'total_rows': int(result.iloc[0]['total_rows']),
                'phishing_count': int(result.iloc[0]['phishing_count']),
                'legitimate_count': int(result.iloc[0]['legitimate_count']),
                'database': database_name,
                'table': table_name
            }
            
            if info['total_rows'] > 0:
                info['phishing_ratio'] = info['phishing_count'] / info['total_rows']
            
            return info
            
        except Exception as e:
            logging.error(f"获取数据库信息失败: {str(e)}")
            return None