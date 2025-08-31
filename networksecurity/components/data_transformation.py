import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod  # 添加静态方法装饰器
    def get_data_transformer_object() -> Pipeline:  # 修复方法签名
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Returns:
          A Pipeline object
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"  # 修复拼写错误
        )
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialised KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            
            # 验证输入文件是否存在
            if not os.path.exists(self.data_validation_artifact.valid_train_file_path):
                raise FileNotFoundError(f"Training file not found: {self.data_validation_artifact.valid_train_file_path}")
            
            if not os.path.exists(self.data_validation_artifact.valid_test_file_path):
                raise FileNotFoundError(f"Testing file not found: {self.data_validation_artifact.valid_test_file_path}")
            
            # 读取验证后的数据
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Testing data shape: {test_df.shape}")

            # 处理训练数据
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)  # 将-1转换为0
            
            logging.info(f"Training features shape: {input_feature_train_df.shape}")
            logging.info(f"Training target distribution: {target_feature_train_df.value_counts().to_dict()}")

            # 处理测试数据
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)  # 将-1转换为0
            
            logging.info(f"Testing features shape: {input_feature_test_df.shape}")
            logging.info(f"Testing target distribution: {target_feature_test_df.value_counts().to_dict()}")

            # 获取预处理管道
            preprocessor = self.get_data_transformer_object()

            # 训练预处理器
            logging.info("Fitting preprocessor on training data")
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            
            # 转换训练和测试数据
            logging.info("Transforming training data")
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            
            logging.info("Transforming testing data")
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            logging.info(f"Transformed training data shape: {transformed_input_train_feature.shape}")
            logging.info(f"Transformed testing data shape: {transformed_input_test_feature.shape}")

            # 合并特征和目标变量
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            logging.info(f"Final training array shape: {train_arr.shape}")
            logging.info(f"Final testing array shape: {test_arr.shape}")

            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            # 保存转换后的numpy数组数据
            logging.info("Saving transformed data arrays")
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, 
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            
            # 保存预处理对象
            logging.info("Saving preprocessor object")
            save_object(
                self.data_transformation_config.transformed_object_file_path, 
                preprocessor_object
            )

            # 也保存到final_model目录（确保目录存在）
            final_model_dir = "final_model"
            os.makedirs(final_model_dir, exist_ok=True)
            save_object(
                os.path.join(final_model_dir, "preprocessor.pkl"), 
                preprocessor_object
            )

            logging.info("Data transformation completed successfully")

            # 创建artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            logging.info(f"Data transformation artifact created:")
            logging.info(f"  - Preprocessor: {data_transformation_artifact.transformed_object_file_path}")
            logging.info(f"  - Training data: {data_transformation_artifact.transformed_train_file_path}")
            logging.info(f"  - Testing data: {data_transformation_artifact.transformed_test_file_path}")
            
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)