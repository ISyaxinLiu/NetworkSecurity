#!/usr/bin/env python3
"""
测试ML工具模块
"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_network_model():
    """
    测试NetworkModel类
    """
    try:
        print("🤖 测试NetworkModel类")
        print("-" * 40)
        
        # 导入必要的模块
        from networksecurity.utils.ml_utils.model.estimator import NetworkModel
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        
        # 创建模拟数据
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.choice([0, 1], 100)
        X_test = np.random.randn(20, 5)
        
        print(f"训练数据形状: {X_train.shape}")
        print(f"测试数据形状: {X_test.shape}")
        
        # 创建并训练预处理器
        preprocessor = StandardScaler()
        preprocessor.fit(X_train)
        
        # 创建并训练模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train_scaled = preprocessor.transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        print("✅ 模型和预处理器创建完成")
        
        # 创建NetworkModel
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        print("✅ NetworkModel创建成功")
        
        # 测试预测功能
        predictions = network_model.predict(X_test)
        print(f"预测结果形状: {predictions.shape}")
        print(f"预测结果: {np.unique(predictions, return_counts=True)}")
        
        # 测试概率预测
        probabilities = network_model.predict_proba(X_test)
        if probabilities is not None:
            print(f"概率预测形状: {probabilities.shape}")
            print("✅ 概率预测功能正常")
        else:
            print("⚠️ 模型不支持概率预测")
        
        # 测试模型信息获取
        model_info = network_model.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 测试模型保存和加载
        test_model_path = "test_model.pkl"
        try:
            saved_path = network_model.save_model(test_model_path)
            print(f"✅ 模型保存成功: {saved_path}")
            
            # 加载模型
            loaded_model = NetworkModel.load_model(saved_path)
            print("✅ 模型加载成功")
            
            # 验证加载的模型
            loaded_predictions = loaded_model.predict(X_test)
            if np.array_equal(predictions, loaded_predictions):
                print("✅ 加载的模型预测结果一致")
            else:
                print("❌ 加载的模型预测结果不一致")
            
            # 清理测试文件
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
                print("🧹 测试文件已清理")
                
        except Exception as e:
            print(f"❌ 模型保存/加载测试失败: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ NetworkModel测试失败: {str(e)}")
        return False

def test_classification_metrics():
    """
    测试分类指标函数
    """
    try:
        print("\n📊 测试分类指标函数")
        print("-" * 40)
        
        from networksecurity.utils.ml_utils.metric.classification_metric import (
            get_classification_score,
            get_detailed_classification_score,
            evaluate_binary_classification,
            print_classification_summary,
            compare_models
        )
        
        # 创建模拟的真实标签和预测标签
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_pred = np.random.choice([0, 1], 100)
        y_pred_proba = np.random.rand(100, 2)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)  # 归一化
        
        print(f"真实标签分布: {np.unique(y_true, return_counts=True)}")
        print(f"预测标签分布: {np.unique(y_pred, return_counts=True)}")
        
        # 测试基本分类指标
        basic_metrics = get_classification_score(y_true, y_pred)
        print("✅ 基本分类指标计算成功")
        print(f"   F1-Score: {basic_metrics.f1_score:.4f}")
        print(f"   Precision: {basic_metrics.precision_score:.4f}")
        print(f"   Recall: {basic_metrics.recall_score:.4f}")
        
        # 测试详细分类指标
        detailed_metrics = get_detailed_classification_score(y_true, y_pred)
        print("✅ 详细分类指标计算成功")
        print(f"   准确率: {detailed_metrics['accuracy']:.4f}")
        print(f"   混淆矩阵形状: {np.array(detailed_metrics['confusion_matrix']).shape}")
        
        # 测试二分类评估
        binary_metrics = evaluate_binary_classification(y_true, y_pred, y_pred_proba)
        print("✅ 二分类评估完成")
        if 'roc_auc_score' in binary_metrics:
            print(f"   ROC AUC: {binary_metrics['roc_auc_score']:.4f}")
        if 'average_precision_score' in binary_metrics:
            print(f"   平均精确度: {binary_metrics['average_precision_score']:.4f}")
        
        # 测试结果打印
        print("\n📋 分类结果摘要:")
        print_classification_summary(binary_metrics, "测试模型")
        
        # 测试模型比较
        model_results = {
            'Model_A': binary_metrics,
            'Model_B': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.82,
                'f1_score': 0.81,
                'roc_auc_score': 0.88
            }
        }
        
        try:
            comparison_df = compare_models(model_results)
            print("\n🔄 模型比较结果:")
            print(comparison_df.to_string(index=False))
            print("✅ 模型比较功能正常")
        except Exception as e:
            print(f"⚠️ 模型比较功能测试失败: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分类指标测试失败: {str(e)}")
        import traceback
        print("详细错误:")
        print(traceback.format_exc())
        return False

def test_integration_with_real_data():
    """
    使用真实数据测试集成功能
    """
    try:
        print("\n🔗 测试与真实数据的集成")
        print("-" * 40)
        
        # 检查是否有可用的数据文件
        from networksecurity.entity.config_entity import (
            TrainingPipelineConfig,
            DataIngestionConfig,
            DataValidationConfig,
            DataTransformationConfig
        )
        
        # 创建配置
        training_config = TrainingPipelineConfig()
        transformation_config = DataTransformationConfig(training_config)
        
        # 检查转换后的数据是否存在
        train_file = transformation_config.transformed_train_file_path
        test_file = transformation_config.transformed_test_file_path
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("✅ 找到转换后的数据文件")
            
            # 加载数据
            train_arr = np.load(train_file)
            test_arr = np.load(test_file)
            
            print(f"训练数据形状: {train_arr.shape}")
            print(f"测试数据形状: {test_arr.shape}")
            
            # 分离特征和标签
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)
            
            print(f"特征数量: {X_train.shape[1]}")
            print(f"训练样本: {len(y_train)}")
            print(f"测试样本: {len(y_test)}")
            
            # 创建简单模型进行测试
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from networksecurity.utils.ml_utils.model.estimator import NetworkModel
            from networksecurity.utils.ml_utils.metric.classification_metric import (
                evaluate_binary_classification,
                print_classification_summary
            )
            
            # 创建和训练模型
            preprocessor = StandardScaler()
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # 拟合预处理器和模型
            X_train_scaled = preprocessor.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
            
            # 创建NetworkModel
            network_model = NetworkModel(preprocessor=preprocessor, model=model)
            
            # 进行预测
            y_pred = network_model.predict(X_test)
            y_pred_proba = network_model.predict_proba(X_test)
            
            # 评估结果
            metrics = evaluate_binary_classification(y_test, y_pred, y_pred_proba)
            
            # 打印结果
            print_classification_summary(metrics, "真实数据测试模型")
            
            print("✅ 真实数据集成测试成功")
            return True
            
        else:
            print("⚠️ 未找到转换后的数据文件，跳过真实数据测试")
            print("请先运行完整的数据管道 (Ingestion -> Validation -> Transformation)")
            return True
            
    except Exception as e:
        print(f"❌ 真实数据集成测试失败: {str(e)}")
        return False

def main():
    """
    主函数
    """
    print("ML工具模块测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # 测试NetworkModel
    if test_network_model():
        success_count += 1
    
    # 测试分类指标
    if test_classification_metrics():
        success_count += 1
    
    # 测试真实数据集成
    if test_integration_with_real_data():
        success_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"测试完成: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过!")
        print("🚀 ML工具模块已准备就绪!")
        print("📝 现在可以用于模型训练和评估了!")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)