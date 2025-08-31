from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging 
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        验证数据框的列数是否符合schema定义
        """
        try:
            # 修复：从schema中正确获取列数
            if 'columns' in self._schema_config:
                expected_columns = len(self._schema_config['columns'])
            else:
                # 兼容旧版schema格式
                expected_columns = len(self._schema_config)
                
            actual_columns = len(dataframe.columns)
            
            logging.info(f"Required number of columns: {expected_columns}")
            logging.info(f"Data frame has columns: {actual_columns}")
            logging.info(f"Data frame columns: {list(dataframe.columns)}")
            
            if actual_columns == expected_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        验证数值列是否存在且类型正确
        """
        try:
            if 'numerical_columns' not in self._schema_config:
                logging.warning("Schema中未定义numerical_columns，跳过数值列验证")
                return True
            
            numerical_columns = self._schema_config['numerical_columns']
            dataframe_columns = dataframe.columns
            
            missing_columns = []
            for column in numerical_columns:
                if column not in dataframe_columns:
                    missing_columns.append(column)
            
            if missing_columns:
                logging.error(f"Missing numerical columns: {missing_columns}")
                return False
            
            # 检查数据类型
            for column in numerical_columns:
                if column in dataframe.columns:
                    if not pd.api.types.is_numeric_dtype(dataframe[column]):
                        logging.error(f"Column {column} is not numeric type")
                        return False
            
            logging.info("所有数值列验证通过")
            return True
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_target_column(self, dataframe: pd.DataFrame) -> bool:
        """
        验证目标列是否存在且值符合要求
        """
        try:
            target_column = self._schema_config.get('target_column', 'Result')
            
            if target_column not in dataframe.columns:
                logging.error(f"Target column {target_column} not found in dataframe")
                return False
            
            # 检查目标列的值
            unique_values = dataframe[target_column].unique()
            expected_values = [-1, 1]  # 钓鱼检测的标签值
            
            for value in unique_values:
                if value not in expected_values:
                    logging.error(f"Unexpected value in target column: {value}")
                    return False
            
            logging.info(f"Target column validation passed. Values: {unique_values}")
            return True
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        检测数据集漂移
        """
        try:
            status = True
            report = {}
            
            # 只检查数值列的漂移
            numerical_columns = self._schema_config.get('numerical_columns', base_df.columns)
            
            for column in numerical_columns:
                if column in base_df.columns and column in current_df.columns:
                    d1 = base_df[column]
                    d2 = current_df[column]
                    
                    # 执行KS检验
                    is_same_dist = ks_2samp(d1, d2)
                    
                    if threshold <= is_same_dist.pvalue:
                        is_found = False  # 没有发现漂移
                    else:
                        is_found = True   # 发现漂移
                        status = False    # 整体状态设为False
                    
                    report.update({column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found,
                        "threshold": threshold
                    }})
                    
                    logging.info(f"Column {column}: p_value={is_same_dist.pvalue:.4f}, drift={is_found}")
            
            # 保存漂移报告
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            # 创建目录
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # 添加汇总信息
            report['summary'] = {
                'total_columns_checked': len(report) - 1 if 'summary' not in report else len(report),
                'drift_detected': not status,
                'threshold_used': threshold,
                'drift_columns': [col for col, info in report.items() 
                                if isinstance(info, dict) and info.get('drift_status', False)]
            }
            
            write_yaml_file(file_path=drift_report_file_path, content=report)
            logging.info(f"Drift report saved to: {drift_report_file_path}")
            
            return status
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        启动数据验证流程
        """
        try:
            error_message = ""
            validation_status = True
            
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("开始数据验证流程")
            logging.info(f"训练文件: {train_file_path}")
            logging.info(f"测试文件: {test_file_path}")

            # 读取训练和测试数据
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            logging.info(f"训练数据形状: {train_dataframe.shape}")
            logging.info(f"测试数据形状: {test_dataframe.shape}")

            # 验证训练数据列数
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                validation_status = False
                error_message += "Train dataframe does not contain all columns.\n"
                logging.error("训练数据列数验证失败")
            
            # 验证测试数据列数
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                validation_status = False
                error_message += "Test dataframe does not contain all columns.\n"
                logging.error("测试数据列数验证失败")
            
            # 验证数值列
            if validation_status:
                status = self.validate_numerical_columns(dataframe=train_dataframe)
                if not status:
                    validation_status = False
                    error_message += "Train dataframe numerical columns validation failed.\n"
                
                status = self.validate_numerical_columns(dataframe=test_dataframe)
                if not status:
                    validation_status = False
                    error_message += "Test dataframe numerical columns validation failed.\n"
            
            # 验证目标列
            if validation_status:
                status = self.validate_target_column(dataframe=train_dataframe)
                if not status:
                    validation_status = False
                    error_message += "Train dataframe target column validation failed.\n"
                
                status = self.validate_target_column(dataframe=test_dataframe)
                if not status:
                    validation_status = False
                    error_message += "Test dataframe target column validation failed.\n"

            # 检测数据漂移（即使前面验证失败也要执行）
            logging.info("开始检测数据漂移")
            drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            
            if not drift_status:
                logging.warning("检测到数据漂移")
            else:
                logging.info("未检测到显著数据漂移")

            # 准备输出目录
            valid_dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            invalid_dir_path = os.path.dirname(self.data_validation_config.invalid_train_file_path)
            
            os.makedirs(valid_dir_path, exist_ok=True)
            os.makedirs(invalid_dir_path, exist_ok=True)

            # 根据验证结果保存文件
            if validation_status:
                # 验证通过，保存到valid目录
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path, 
                    index=False, header=True
                )
                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path, 
                    index=False, header=True
                )
                
                logging.info("数据验证通过，文件保存到valid目录")
                
                data_validation_artifact = DataValidationArtifact(
                    validation_status=validation_status,
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )
            else:
                # 验证失败，保存到invalid目录
                train_dataframe.to_csv(
                    self.data_validation_config.invalid_train_file_path, 
                    index=False, header=True
                )
                test_dataframe.to_csv(
                    self.data_validation_config.invalid_test_file_path, 
                    index=False, header=True
                )
                
                logging.error(f"数据验证失败: {error_message}")
                
                data_validation_artifact = DataValidationArtifact(
                    validation_status=validation_status,
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )

            logging.info("数据验证流程完成")
            return data_validation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)