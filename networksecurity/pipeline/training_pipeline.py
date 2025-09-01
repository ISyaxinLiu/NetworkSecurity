import os
import sys
import shutil
from datetime import datetime

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, 
                                            data_validation_config=data_validation_config)
            logging.info("Initiate the data Validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed: {data_validation_artifact.validation_status}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                    data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def backup_artifacts_locally(self):
        """
        本地备份工件 - 替代S3同步
        """
        try:
            # 创建备份目录
            backup_base_dir = "model_backups"
            os.makedirs(backup_base_dir, exist_ok=True)
            
            timestamp = self.training_pipeline_config.timestamp
            backup_dir = os.path.join(backup_base_dir, f"backup_{timestamp}")
            
            # 备份工件目录
            if os.path.exists(self.training_pipeline_config.artifact_dir):
                artifact_backup_dir = os.path.join(backup_dir, "artifacts")
                shutil.copytree(self.training_pipeline_config.artifact_dir, artifact_backup_dir)
                logging.info(f"Artifacts backed up to: {artifact_backup_dir}")
            
            # 备份final_model目录
            final_model_dir = "final_model"
            if os.path.exists(final_model_dir):
                model_backup_dir = os.path.join(backup_dir, "final_model")
                shutil.copytree(final_model_dir, model_backup_dir)
                logging.info(f"Final model backed up to: {model_backup_dir}")
            
            # 创建备份信息文件
            backup_info_path = os.path.join(backup_dir, "backup_info.txt")
            with open(backup_info_path, 'w') as f:
                f.write(f"备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"管道时间戳: {timestamp}\n")
                f.write(f"备份目录: {backup_dir}\n")
            
            logging.info(f"本地备份完成: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logging.warning(f"本地备份失败: {str(e)}")
            raise NetworkSecurityException(e, sys)
    
    def save_model_summary(self, model_trainer_artifact: ModelTrainerArtifact):
        """
        保存模型训练总结
        """
        try:
            summary_dir = "model_summaries"
            os.makedirs(summary_dir, exist_ok=True)
            
            summary_file = os.path.join(summary_dir, f"model_summary_{self.training_pipeline_config.timestamp}.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("网络安全钓鱼检测模型训练总结\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"管道时间戳: {self.training_pipeline_config.timestamp}\n\n")
                
                f.write("模型性能指标:\n")
                f.write("-" * 20 + "\n")
                
                train_metrics = model_trainer_artifact.train_metric_artifact
                test_metrics = model_trainer_artifact.test_metric_artifact
                
                f.write(f"训练集性能:\n")
                f.write(f"  - F1分数: {train_metrics.f1_score:.4f}\n")
                f.write(f"  - 精确率: {train_metrics.precision_score:.4f}\n")
                f.write(f"  - 召回率: {train_metrics.recall_score:.4f}\n\n")
                
                f.write(f"测试集性能:\n")
                f.write(f"  - F1分数: {test_metrics.f1_score:.4f}\n")
                f.write(f"  - 精确率: {test_metrics.precision_score:.4f}\n")
                f.write(f"  - 召回率: {test_metrics.recall_score:.4f}\n\n")
                
                performance_diff = abs(train_metrics.f1_score - test_metrics.f1_score)
                f.write(f"泛化性能:\n")
                f.write(f"  - 训练测试F1差异: {performance_diff:.4f}\n")
                
                if performance_diff < 0.05:
                    f.write("  - 泛化性能: 良好\n")
                else:
                    f.write("  - 泛化性能: 需要关注\n")
                
                f.write(f"\n模型文件:\n")
                f.write(f"  - 训练模型: {model_trainer_artifact.trained_model_file_path}\n")
                f.write(f"  - Final模型: final_model/model.pkl\n")
            
            logging.info(f"模型总结保存到: {summary_file}")
            return summary_file
            
        except Exception as e:
            logging.warning(f"保存模型总结失败: {str(e)}")
    
    def cleanup_old_artifacts(self, keep_last_n: int = 5):
        """
        清理旧的工件，只保留最近的N个
        """
        try:
            artifacts_base_dir = "Artifacts"
            if not os.path.exists(artifacts_base_dir):
                return
            
            # 获取所有工件目录
            artifact_dirs = []
            for item in os.listdir(artifacts_base_dir):
                item_path = os.path.join(artifacts_base_dir, item)
                if os.path.isdir(item_path):
                    try:
                        # 解析时间戳
                        timestamp = datetime.strptime(item, "%m_%d_%Y_%H_%M_%S")
                        artifact_dirs.append((timestamp, item_path))
                    except ValueError:
                        continue
            
            # 按时间排序
            artifact_dirs.sort(key=lambda x: x[0], reverse=True)
            
            # 删除旧的工件
            for i, (timestamp, path) in enumerate(artifact_dirs):
                if i >= keep_last_n:
                    shutil.rmtree(path)
                    logging.info(f"删除旧工件: {path}")
            
            logging.info(f"工件清理完成，保留最近 {keep_last_n} 个")
            
        except Exception as e:
            logging.warning(f"清理旧工件失败: {str(e)}")
    
    def run_pipeline(self):
        try:
            logging.info("开始运行训练管道")
            
            # 执行管道各个阶段
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            
            # 检查数据验证是否通过
            if not data_validation_artifact.validation_status:
                raise Exception("数据验证失败，停止管道执行")
            
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            
            # 本地操作替代云同步
            logging.info("开始本地备份和总结")
            
            # 本地备份
            backup_dir = self.backup_artifacts_locally()
            
            # 保存模型总结
            summary_file = self.save_model_summary(model_trainer_artifact)
            
            # 清理旧工件
            self.cleanup_old_artifacts(keep_last_n=3)
            
            # 显示最终结果
            logging.info("训练管道执行完成")
            logging.info(f"模型性能 - 测试F1: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
            logging.info(f"本地备份: {backup_dir}")
            
            if summary_file:
                logging.info(f"模型总结: {summary_file}")
            
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)


# 使用示例
if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        model_artifact = pipeline.run_pipeline()
        
        print("管道执行完成!")
        print(f"最终模型性能:")
        print(f"  测试集F1分数: {model_artifact.test_metric_artifact.f1_score:.4f}")
        print(f"  测试集精确率: {model_artifact.test_metric_artifact.precision_score:.4f}")
        print(f"  测试集召回率: {model_artifact.test_metric_artifact.recall_score:.4f}")
        
    except Exception as e:
        print(f"管道执行失败: {str(e)}")