import os
import sys
import dagshub
import mlflow
from datetime import datetime  # 添加这行

# 在所有其他导入之前设置MLflow配置
def setup_mlflow_tracking():
    """设置MLflow跟踪配置"""
    print("正在设置MLflow配置...")
    
    # 设置环境变量
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "ISyaxinLiu"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "f36ccefd8aa76fa3d07a50e0baf446776f28f379"
    
    # 初始化DagHub
    try:
        dagshub.init(repo_owner='ISyaxinLiu', repo_name='NetworkSecurity', mlflow=True)
        print("DagHub初始化成功")
        
        # 设置实验
        experiment_name = "NetworkSecurity_Pipeline"
        try:
            mlflow.set_experiment(experiment_name)
            print(f"设置实验: {experiment_name}")
        except:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"创建并设置实验: {experiment_name}")
            
        return True
        
    except Exception as e:
        print(f"MLflow设置失败: {str(e)}")
        return False

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import ModelTrainerConfig
import os
import dagshub

# 在导入后立即设置MLflow配置
def setup_mlflow():
    """
    设置MLflow配置
    """
    # 设置你的MLflow配置
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "ISyaxinLiu"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "f36ccefd8aa76fa3d07a50e0baf446776f28f379"
    
    # 初始化DagHub - 这很重要！
    try:
        dagshub.init(repo_owner='ISyaxinLiu', repo_name='NetworkSecurity', mlflow=True)
        logging.info("DagHub和MLflow初始化成功")
        print("MLflow配置完成")
    except Exception as e:
        logging.warning(f"DagHub初始化失败: {str(e)}")
        print(f"MLflow配置可能有问题: {str(e)}")

if __name__ == '__main__':
    try:
        # 首先设置MLflow配置
        print("开始设置MLflow配置...")
        mlflow_setup_success = setup_mlflow()
        
        if not mlflow_setup_success:
            print("MLflow设置失败，将继续运行但不记录实验")
        
        print("开始执行机器学习管道...")
        
        # 创建训练管道配置
        trainingpipelineconfig = TrainingPipelineConfig()
        
        # 1. 数据摄取
        logging.info("=== 开始数据摄取阶段 ===")
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print("数据摄取完成:", dataingestionartifact)
        
        # 2. 数据验证
        logging.info("=== 开始数据验证阶段 ===")
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print("数据验证完成:", data_validation_artifact)
        
        # 检查验证是否通过
        if not data_validation_artifact.validation_status:
            raise Exception("数据验证失败，停止管道执行")
        
        # 3. 数据转换
        logging.info("=== 开始数据转换阶段 ===")
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print("数据转换完成:", data_transformation_artifact)
        logging.info("data Transformation completed")

        # 4. 模型训练
        logging.info("=== 开始模型训练阶段 ===")
        print("\n开始模型训练...")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        
        # 启动模型训练
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")
        print("模型训练完成:", model_trainer_artifact)
        
        # 显示最终结果
        print("\n" + "="*60)
        print("机器学习管道执行完成!")
        print("="*60)
        
        print(f"最终模型性能:")
        print(f"  - 训练集F1: {model_trainer_artifact.train_metric_artifact.f1_score:.4f}")
        print(f"  - 测试集F1: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
        print(f"  - 训练集精确率: {model_trainer_artifact.train_metric_artifact.precision_score:.4f}")
        print(f"  - 测试集精确率: {model_trainer_artifact.test_metric_artifact.precision_score:.4f}")
        
        print(f"\n保存的文件:")
        print(f"  - 训练模型: {model_trainer_artifact.trained_model_file_path}")
        print(f"  - Final model: final_model/model.pkl")
        
        # 显示MLflow链接（如果设置成功）
        if mlflow_setup_success:
            print(f"\nMLflow实验链接:")
            print(f"  https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow")
            
            # 尝试获取最近的运行信息
            try:
                client = mlflow.tracking.MlflowClient()
                experiment = mlflow.get_experiment_by_name("NetworkSecurity_Pipeline")
                if experiment:
                    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
                    if runs:
                        latest_run = runs[0]
                        print(f"  最新运行ID: {latest_run.info.run_id}")
                        print(f"  运行状态: {latest_run.info.status}")
                        print(f"  直接链接: https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow/#/experiments/{experiment.experiment_id}/runs/{latest_run.info.run_id}")
            except Exception as e:
                print(f"获取运行信息失败: {e}")
        
    except Exception as e:
        logging.error(f"管道执行失败: {str(e)}")
        raise NetworkSecurityException(e, sys)