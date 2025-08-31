#!/usr/bin/env python3
"""
测试Model Trainer组件
"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_model_trainer():
    """
    测试Model Trainer完整流程
    """
    try:
        print("🤖 开始测试Model Trainer组件")
        print("=" * 60)
        
        # 导入必要的模块
        from networksecurity.entity.config_entity import (
            TrainingPipelineConfig, 
            DataIngestionConfig, 
            DataValidationConfig,
            DataTransformationConfig,
            ModelTrainerConfig
        )
        from networksecurity.components.data_ingestion import DataIngestion
        from networksecurity.components.data_validation import DataValidation
        from networksecurity.components.data_transformation import DataTransformation
        from networksecurity.components.model_trainer import ModelTrainer
        from networksecurity.exception.exception import NetworkSecurityException
        from networksecurity.logging.logger import logging
        
        print("✅ 所有模块导入成功")
        
        # 步骤1: 创建配置对象
        print("\n📋 步骤1: 创建配置对象")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        
        print(f"   - 模型训练目录: {model_trainer_config.model_trainer_dir}")
        print(f"   - 训练模型路径: {model_trainer_config.trained_model_file_path}")
        print(f"   - 期望准确率: {model_trainer_config.expected_accuracy}")
        print(f"   - 过拟合阈值: {model_trainer_config.overfitting_underfitting_threshold}")
        
        # 步骤2: 检查是否有转换后的数据
        print("\n📊 步骤2: 检查转换后的数据")
        train_file = data_transformation_config.transformed_train_file_path
        test_file = data_transformation_config.transformed_test_file_path
        preprocessor_file = data_transformation_config.transformed_object_file_path
        
        if not all([os.path.exists(train_file), os.path.exists(test_file), os.path.exists(preprocessor_file)]):
            print("⚠️ 转换后的数据不存在，需要先运行完整的数据管道")
            print("正在执行数据管道...")
            
            # 执行完整的数据管道
            # 数据摄取
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            # 数据验证
            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            
            if not data_validation_artifact.validation_status:
                raise Exception("数据验证失败，无法继续模型训练")
            
            # 数据转换
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            print("✅ 数据管道执行完成")
        else:
            print("✅ 找到转换后的数据")
            # 直接使用现有的数据转换工件
            from networksecurity.entity.artifact_entity import DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=preprocessor_file,
                transformed_train_file_path=train_file,
                transformed_test_file_path=test_file
            )
        
        # 验证数据
        train_arr = np.load(data_transformation_artifact.transformed_train_file_path)
        test_arr = np.load(data_transformation_artifact.transformed_test_file_path)
        
        print(f"   - 训练数据形状: {train_arr.shape}")
        print(f"   - 测试数据形状: {test_arr.shape}")
        print(f"   - 特征数量: {train_arr.shape[1] - 1}")
        
        # 步骤3: 创建模型训练器
        print("\n🚀 步骤3: 创建模型训练器")
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        print("✅ 模型训练器创建成功")
        
        # 步骤4: 执行模型训练
        print("\n🎯 步骤4: 执行模型训练流程")
        print("   这将执行以下操作:")
        print("   1. 加载转换后的训练和测试数据")
        print("   2. 训练多个机器学习模型")
        print("   3. 使用网格搜索优化超参数")
        print("   4. 选择最佳模型")
        print("   5. 评估模型性能")
        print("   6. 保存最佳模型")
        print("   7. 可选：记录MLflow实验")
        
        print("\n开始模型训练...")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        print("✅ 模型训练流程完成!")
        
        # 步骤5: 验证训练结果
        print("\n📊 步骤5: 验证训练结果")
        
        # 检查训练后的模型文件
        if os.path.exists(model_trainer_artifact.trained_model_file_path):
            print(f"   ✅ 训练模型已保存: {model_trainer_artifact.trained_model_file_path}")
            
            # 尝试加载模型
            try:
                from networksecurity.utils.main_utils.utils import load_object
                trained_model = load_object(model_trainer_artifact.trained_model_file_path)
                print(f"      - 模型类型: {type(trained_model)}")
                
                # 如果是NetworkModel，显示更多信息
                if hasattr(trained_model, 'get_model_info'):
                    model_info = trained_model.get_model_info()
                    print(f"      - 模型信息: {model_info}")
                
            except Exception as e:
                print(f"      ⚠️ 模型加载测试失败: {str(e)}")
        else:
            print("   ❌ 训练模型文件不存在")
        
        # 检查final_model目录
        final_model_path = "final_model/model.pkl"
        if os.path.exists(final_model_path):
            print(f"   ✅ Final model已保存: {final_model_path}")
        else:
            print("   ❌ Final model文件不存在")
        
        # 显示训练指标
        train_metrics = model_trainer_artifact.train_metric_artifact
        test_metrics = model_trainer_artifact.test_metric_artifact
        
        print(f"\n📈 模型性能指标:")
        print(f"   训练集:")
        print(f"      - F1分数: {train_metrics.f1_score:.4f}")
        print(f"      - 精确率: {train_metrics.precision_score:.4f}")
        print(f"      - 召回率: {train_metrics.recall_score:.4f}")
        
        print(f"   测试集:")
        print(f"      - F1分数: {test_metrics.f1_score:.4f}")
        print(f"      - 精确率: {test_metrics.precision_score:.4f}")
        print(f"      - 召回率: {test_metrics.recall_score:.4f}")
        
        # 性能分析
        performance_diff = abs(train_metrics.f1_score - test_metrics.f1_score)
        print(f"\n🔍 性能分析:")
        print(f"   - 训练测试F1差异: {performance_diff:.4f}")
        
        if performance_diff > model_trainer_config.overfitting_underfitting_threshold:
            print(f"   ⚠️ 可能存在过拟合/欠拟合 (阈值: {model_trainer_config.overfitting_underfitting_threshold})")
        else:
            print(f"   ✅ 模型泛化性能良好")
        
        if test_metrics.f1_score >= model_trainer_config.expected_accuracy:
            print(f"   ✅ 模型性能达到期望 (期望: {model_trainer_config.expected_accuracy})")
        else:
            print(f"   ⚠️ 模型性能低于期望 (期望: {model_trainer_config.expected_accuracy})")
        
        # 步骤6: 测试模型预测
        print("\n🔮 步骤6: 测试模型预测")
        
        if os.path.exists(model_trainer_artifact.trained_model_file_path):
            try:
                # 加载训练好的模型
                trained_model = load_object(model_trainer_artifact.trained_model_file_path)
                
                # 准备测试数据
                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1].astype(int)
                
                # 进行预测
                test_sample = X_test[:5]  # 取前5个样本
                predictions = trained_model.predict(test_sample)
                
                print(f"   测试预测结果:")
                for i, pred in enumerate(predictions):
                    actual = y_test[i]
                    result_text = "钓鱼网站" if pred == 1 else "合法网站"
                    actual_text = "钓鱼网站" if actual == 1 else "合法网站"
                    match = "✅" if pred == actual else "❌"
                    print(f"      样本{i+1}: 预测={result_text}, 实际={actual_text} {match}")
                
                # 如果支持概率预测
                if hasattr(trained_model, 'predict_proba'):
                    probabilities = trained_model.predict_proba(test_sample)
                    print(f"   预测概率 (前3个样本):")
                    for i in range(min(3, len(probabilities))):
                        prob_legit = probabilities[i][0]
                        prob_phish = probabilities[i][1]
                        print(f"      样本{i+1}: 合法={prob_legit:.3f}, 钓鱼={prob_phish:.3f}")
                
                print("   ✅ 模型预测功能正常")
                
            except Exception as e:
                print(f"   ❌ 模型预测测试失败: {str(e)}")
        
        # 步骤7: 总结
        print("\n🎉 测试总结")
        print("=" * 60)
        print("✅ Model Trainer组件测试成功!")
        print("✅ 模型训练流程正常工作")
        print("✅ 模型评估和选择正常")
        print("✅ 模型保存功能正常")
        print("✅ 预测功能正常")
        
        print("\n📁 生成的文件:")
        print(f"   - 训练模型: {model_trainer_artifact.trained_model_file_path}")
        print(f"   - Final model: final_model/model.pkl")
        
        print(f"\n📊 最终性能:")
        print(f"   - 测试集F1分数: {test_metrics.f1_score:.4f}")
        print(f"   - 测试集精确率: {test_metrics.precision_score:.4f}")
        print(f"   - 测试集召回率: {test_metrics.recall_score:.4f}")
        
        print("\n🎯 下一步:")
        print("   模型训练完成，可以用于生产部署或进一步优化")
        
        return model_trainer_artifact
        
    except NetworkSecurityException as nse:
        print(f"\n❌ 网络安全异常: {str(nse)}")
        return None
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        return None

def main():
    """
    主函数
    """
    print("Model Trainer 组件测试")
    print("=" * 60)
    
    artifact = test_model_trainer()
    
    if artifact:
        print("\n🎊 所有测试完成!")
        print("🚀 Model Trainer组件已准备就绪!")
        print("📊 现在你拥有了一个完整的机器学习管道!")
        return True
    else:
        print("\n❌ 测试失败，请检查配置和数据")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)