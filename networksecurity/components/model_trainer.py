import os
import sys
import numpy as np
from datetime import datetime

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_classification_models
# 新增导入：并行优化版本
from networksecurity.utils.main_utils.utils import evaluate_classification_models_parallel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='ISyaxinLiu', repo_name='NetworkSecurity', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def track_mlflow(self, best_model, classificationmetric, model_name: str, additional_metrics=None):
        """
        使用MLflow追踪模型实验 - 增强版本
        """
        print(f"开始MLflow跟踪: {model_name}")
        
        try:
            # 确保没有活跃的运行
            if mlflow.active_run():
                print("结束现有MLflow运行")
                mlflow.end_run()
            
            mlflow.set_registry_uri("https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow")
            print("MLflow registry URI设置完成")
            
            # 设置实验
            experiment_name = "NetworkSecurity_Training_Enhanced"  # 新实验名，区分升级版
            try:
                mlflow.set_experiment(experiment_name)
                print(f"实验设置成功: {experiment_name}")
            except:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                print(f"实验创建并设置成功: {experiment_name}")
            
            # 创建运行
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name) as run:
                print(f"MLflow运行创建成功: {run.info.run_id}")
                
                f1_score = float(classificationmetric.f1_score)
                precision_score = float(classificationmetric.precision_score)
                recall_score = float(classificationmetric.recall_score)
                
                print(f"记录指标: F1={f1_score:.4f}, Precision={precision_score:.4f}, Recall={recall_score:.4f}")

                # 记录基础指标
                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                
                # 记录额外指标（如果有）
                if additional_metrics:
                    for key, value in additional_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"additional_{key}", float(value))
                
                # 记录参数
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("data_source", "postgresql")
                mlflow.log_param("optimization_method", "optuna_parallel")  # 新增标识
                mlflow.log_param("timestamp", datetime.now().isoformat())
                
                # 记录标签
                mlflow.set_tag("project", "NetworkSecurity")
                mlflow.set_tag("task", "phishing_detection")
                mlflow.set_tag("version", "enhanced_v1")  # 版本标识
                
                print("指标和参数记录完成")
                                
                # 记录模型
                try:
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(best_model, "model")  # 移除registered_model_name参数
                        print("模型注册完成")
                    else:
                        mlflow.sklearn.log_model(best_model, "model")
                        print("模型记录完成")
                except Exception as model_error:
                    print(f"模型记录失败: {model_error}")
                
                experiment_id = run.info.experiment_id
                run_id = run.info.run_id
                
                mlflow_url = f"https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow/#/experiments/{experiment_id}/runs/{run_id}"
                
                print(f"MLflow跟踪完成!")
                print(f"实验ID: {experiment_id}")
                print(f"运行ID: {run_id}")
                print(f"查看链接: {mlflow_url}")
                
                # 记录到日志
                logging.info(f"MLflow跟踪完成: {mlflow_url}")
                
        except Exception as e:
            print(f"MLflow跟踪失败: {str(e)}")
            import traceback
            traceback.print_exc()
            logging.warning(f"MLflow tracking failed: {str(e)}")

    def train_model_enhanced(self, X_train, y_train, X_test, y_test):
        """
        🔥 增强版训练方法：Optuna + 并行化
        """
        try:
            logging.info("=== 开始增强版模型训练 ===")
            logging.info("特性：Optuna贝叶斯优化 + 并行训练 + 详细结果记录")
            
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "AdaBoost": AdaBoostClassifier(random_state=42),
            }
            
            # 🚀 扩展的超参数空间（面试加分点）
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 7, 10, 15, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'max_features': ['sqrt', 'log2'],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 4, 5, 6, 7]
                },
                "Logistic Regression": {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'solver': ['liblinear', 'saga', 'lbfgs'],
                    'penalty': ['l1', 'l2', 'none'],
                },
                "AdaBoost": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                    'n_estimators': [50, 100, 200, 300, 500],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            }
            
            logging.info(f"训练数据形状: {X_train.shape}")
            logging.info(f"测试数据形状: {X_test.shape}")
            logging.info(f"训练标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            logging.info(f"测试标签分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
            
            # 🎯 使用并行化Optuna优化（核心升级）
            import time
            start_time = time.time()
            
            model_report: dict = evaluate_classification_models_parallel(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test,
                models=models, param=params,
                use_optuna=True,  # 启用Optuna
                n_trials=30,      # 每个模型30次试验（可调整）
                n_jobs=-1         # 使用所有CPU核心
            )
            
            end_time = time.time()
            optimization_time = end_time - start_time
            logging.info(f"🚀 并行优化完成！耗时: {optimization_time:.2f}秒")
            
            # 🏆 获取最佳模型
            best_model_score = 0
            best_model_name = None
            
            logging.info("=== 模型性能详细报告 ===")
            for model_name, metrics in model_report.items():
                f1_score = metrics['f1_score']
                logging.info(f"{model_name}:")
                logging.info(f"  F1分数: {f1_score:.4f}")
                logging.info(f"  精确率: {metrics['precision_score']:.4f}")
                logging.info(f"  召回率: {metrics['recall_score']:.4f}")
                logging.info(f"  准确率: {metrics['accuracy']:.4f}")
                logging.info(f"  最佳参数: {metrics.get('best_params', {})}")
                if 'n_trials' in metrics:
                    logging.info(f"  优化试验次数: {metrics['n_trials']}")
                logging.info("")
                
                if f1_score > best_model_score:
                    best_model_score = f1_score
                    best_model_name = model_name
            
            logging.info(f"🏆 最佳模型: {best_model_name}, F1分数: {best_model_score:.4f}")
            
            # 🔧 重新训练最佳模型
            best_model = models[best_model_name]
            best_params = model_report[best_model_name].get('best_params', {})
            
            if best_params:
                best_model.set_params(**best_params)
                logging.info(f"应用最佳参数: {best_params}")
            
            best_model.fit(X_train, y_train)
            
            # 📊 计算详细指标
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            
            logging.info(f"📈 最终训练集指标 - F1: {classification_train_metric.f1_score:.4f}, "
                        f"精确率: {classification_train_metric.precision_score:.4f}, "
                        f"召回率: {classification_train_metric.recall_score:.4f}")
            
            logging.info(f"📊 最终测试集指标 - F1: {classification_test_metric.f1_score:.4f}, "
                        f"精确率: {classification_test_metric.precision_score:.4f}, "
                        f"召回率: {classification_test_metric.recall_score:.4f}")
            
            # 🔍 性能检查
            if classification_test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                logging.warning(f"⚠️ 模型F1分数 {classification_test_metric.f1_score:.4f} "
                              f"低于期望阈值 {self.model_trainer_config.expected_accuracy}")
            
            performance_diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if performance_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(f"⚠️ 检测到可能的过拟合/欠拟合，训练测试性能差异: {performance_diff:.4f}")
            else:
                logging.info(f"✅ 模型泛化性能良好，训练测试性能差异: {performance_diff:.4f}")
            
            # 📝 保存详细结果报告
            try:
                import pandas as pd
                results_data = []
                for name, metrics in model_report.items():
                    results_data.append({
                        'Model': name,
                        'F1_Score': metrics['f1_score'],
                        'Precision': metrics['precision_score'],
                        'Recall': metrics['recall_score'],
                        'Accuracy': metrics['accuracy'],
                        'Best_Params': str(metrics.get('best_params', {})),
                        'Is_Best': name == best_model_name
                    })
                
                df_results = pd.DataFrame(results_data)
                df_results = df_results.sort_values('F1_Score', ascending=False)
                
                os.makedirs("model_comparison_results", exist_ok=True)
                csv_path = f"model_comparison_results/enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_results.to_csv(csv_path, index=False)
                logging.info(f"📁 详细结果已保存到: {csv_path}")
                
            except Exception as e:
                logging.warning(f"结果保存失败: {str(e)}")
            
            # 🔄 MLflow追踪
            try:
                additional_metrics = {
                    'optimization_time': optimization_time,
                    'n_models_compared': len(models),
                    'performance_gap': performance_diff
                }
                self.track_mlflow(best_model, classification_test_metric, best_model_name, additional_metrics)
            except Exception as mlflow_error:
                logging.warning(f"MLflow追踪失败: {str(mlflow_error)}")
            
            # 💾 保存模型
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            # 创建输出目录
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            # 创建完整的网络模型
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            
            # 保存完整模型
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            
            # 保存到final_model目录
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model_enhanced.pkl", network_model)
            
            logging.info(f"✅ 增强版模型已保存到: {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"✅ 增强版模型已保存到: final_model/model_enhanced.pkl")
            
            # 创建模型训练工件
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            logging.info("🎉 增强版模型训练完成!")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        原版训练方法 - 保留兼容性
        """
        try:
            logging.info("开始模型训练和评估")
            
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 7, 10],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'n_estimators': [50, 100, 200]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'saga']
                },
                "AdaBoost": {
                    'learning_rate': [0.01, 0.1, 1.0],
                    'n_estimators': [50, 100, 200]
                }
            }
            
            logging.info(f"训练数据形状: {X_train.shape}")
            logging.info(f"测试数据形状: {X_test.shape}")
            logging.info(f"训练标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            logging.info(f"测试标签分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
            
            # 使用原版分类模型评估函数
            model_report: dict = evaluate_classification_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            logging.info(f"模型评估报告: {model_report}")
            
            # 获取最佳模型 (基于F1分数)
            best_model_score = 0
            best_model_name = None
            
            for model_name, metrics in model_report.items():
                f1_score = metrics['f1_score']
                if f1_score > best_model_score:
                    best_model_score = f1_score
                    best_model_name = model_name
            
            logging.info(f"最佳模型: {best_model_name}, F1分数: {best_model_score:.4f}")
            
            # 重新训练最佳模型
            best_model = models[best_model_name]
            best_params = model_report[best_model_name].get('best_params', {})
            
            if best_params:
                best_model.set_params(**best_params)
            
            best_model.fit(X_train, y_train)
            
            # 计算训练和测试指标
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            
            logging.info(f"训练集指标 - F1: {classification_train_metric.f1_score:.4f}, "
                        f"精确率: {classification_train_metric.precision_score:.4f}, "
                        f"召回率: {classification_train_metric.recall_score:.4f}")
            
            logging.info(f"测试集指标 - F1: {classification_test_metric.f1_score:.4f}, "
                        f"精确率: {classification_test_metric.precision_score:.4f}, "
                        f"召回率: {classification_test_metric.recall_score:.4f}")
            
            # 检查模型性能是否满足要求
            if classification_test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                logging.warning(f"模型F1分数 {classification_test_metric.f1_score:.4f} "
                              f"低于期望阈值 {self.model_trainer_config.expected_accuracy}")
            
            # 检查过拟合/欠拟合
            performance_diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if performance_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(f"检测到可能的过拟合/欠拟合，训练测试性能差异: {performance_diff:.4f}")
            
            # MLflow追踪
            try:
                self.track_mlflow(best_model, classification_test_metric, best_model_name)
            except Exception as mlflow_error:
                logging.warning(f"MLflow追踪失败: {str(mlflow_error)}")
            
            # 加载预处理器
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            # 创建输出目录
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            # 创建完整的网络模型
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            
            # 保存完整模型
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            
            # 保存到final_model目录
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", network_model)
            
            logging.info(f"模型已保存到: {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"模型已保存到: final_model/model.pkl")
            
            # 创建模型训练工件
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self, use_enhanced=True) -> ModelTrainerArtifact:
        """
        启动模型训练流程
        
        Args:
            use_enhanced: 是否使用增强版训练（默认True）
        """
        try:
            logging.info("开始模型训练流程")
            
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            logging.info(f"训练数据文件: {train_file_path}")
            logging.info(f"测试数据文件: {test_file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(train_file_path):
                raise FileNotFoundError(f"训练数据文件不存在: {train_file_path}")
            if not os.path.exists(test_file_path):
                raise FileNotFoundError(f"测试数据文件不存在: {test_file_path}")

            # 加载训练和测试数组
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            logging.info(f"训练数组形状: {train_arr.shape}")
            logging.info(f"测试数组形状: {test_arr.shape}")

            # 分离特征和标签
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)
            
            logging.info(f"特征维度: {X_train.shape[1]}")
            logging.info(f"训练样本数: {len(y_train)}")
            logging.info(f"测试样本数: {len(y_test)}")

            # 🚀 选择训练模式
            if use_enhanced:
                logging.info("🔥 使用增强版训练模式（Optuna + 并行化）")
                model_trainer_artifact = self.train_model_enhanced(X_train, y_train, X_test, y_test)
            else:
                logging.info("使用原版训练模式（GridSearch）")
                model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            
            logging.info("模型训练流程完成")
            logging.info(f"模型训练工件: {model_trainer_artifact}")
            
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)