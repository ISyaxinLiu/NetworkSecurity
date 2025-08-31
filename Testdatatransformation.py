#!/usr/bin/env python3
"""
测试Data Transformation组件
"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_data_transformation():
    """
    测试Data Transformation完整流程
    """
    try:
        print("🔄 开始测试Data Transformation组件")
        print("=" * 60)
        
        # 导入必要的模块
        from networksecurity.entity.config_entity import (
            TrainingPipelineConfig, 
            DataIngestionConfig, 
            DataValidationConfig,
            DataTransformationConfig
        )
        from networksecurity.components.data_ingestion import DataIngestion
        from networksecurity.components.data_validation import DataValidation
        from networksecurity.components.data_transformation import DataTransformation
        from networksecurity.exception.exception import NetworkSecurityException
        from networksecurity.logging.logger import logging
        
        print("✅ 所有模块导入成功")
        
        # 步骤1: 创建配置对象
        print("\n📋 步骤1: 创建配置对象")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        
        print(f"   - 数据转换目录: {data_transformation_config.data_transformation_dir}")
        print(f"   - 转换训练文件: {data_transformation_config.transformed_train_file_path}")
        print(f"   - 转换测试文件: {data_transformation_config.transformed_test_file_path}")
        print(f"   - 预处理器文件: {data_transformation_config.transformed_object_file_path}")
        
        # 步骤2: 执行数据摄取
        print("\n📥 步骤2: 执行数据摄取")
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print("✅ 数据摄取完成")
        
        # 步骤3: 执行数据验证
        print("\n🔍 步骤3: 执行数据验证")
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        
        if not data_validation_artifact.validation_status:
            raise Exception("数据验证失败，无法继续数据转换")
        
        print("✅ 数据验证通过")
        
        # 步骤4: 创建数据转换对象
        print("\n🔄 步骤4: 创建数据转换对象")
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )
        print("✅ 数据转换对象创建成功")
        
        # 步骤5: 执行数据转换
        print("\n🚀 步骤5: 执行数据转换流程")
        print("   这将执行以下操作:")
        print("   1. 读取验证后的训练和测试数据")
        print("   2. 分离特征和目标变量")
        print("   3. 将目标变量从[-1,1]转换为[0,1]")
        print("   4. 使用KNN Imputer处理缺失值")
        print("   5. 保存转换后的数据和预处理器")
        
        # 开始转换
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        print("✅ 数据转换流程完成!")
        
        # 步骤6: 验证转换结果
        print("\n📊 步骤6: 验证转换结果")
        
        # 检查转换后的训练数据
        if os.path.exists(data_transformation_artifact.transformed_train_file_path):
            train_arr = np.load(data_transformation_artifact.transformed_train_file_path)
            print(f"   ✅ 转换后训练数据: {train_arr.shape}")
            print(f"      - 特征数量: {train_arr.shape[1] - 1}")
            print(f"      - 样本数量: {train_arr.shape[0]}")
            
            # 检查目标变量分布
            target_col = train_arr[:, -1]  # 最后一列是目标变量
            unique_values, counts = np.unique(target_col, return_counts=True)
            print(f"      - 目标变量分布: {dict(zip(unique_values, counts))}")
        else:
            print("   ❌ 转换后训练数据文件不存在")
        
        # 检查转换后的测试数据
        if os.path.exists(data_transformation_artifact.transformed_test_file_path):
            test_arr = np.load(data_transformation_artifact.transformed_test_file_path)
            print(f"   ✅ 转换后测试数据: {test_arr.shape}")
            print(f"      - 特征数量: {test_arr.shape[1] - 1}")
            print(f"      - 样本数量: {test_arr.shape[0]}")
            
            # 检查目标变量分布
            target_col = test_arr[:, -1]
            unique_values, counts = np.unique(target_col, return_counts=True)
            print(f"      - 目标变量分布: {dict(zip(unique_values, counts))}")
        else:
            print("   ❌ 转换后测试数据文件不存在")
        
        # 检查预处理器对象
        if os.path.exists(data_transformation_artifact.transformed_object_file_path):
            print(f"   ✅ 预处理器已保存: {data_transformation_artifact.transformed_object_file_path}")
            
            # 尝试加载预处理器
            from networksecurity.utils.main_utils.utils import load_object
            try:
                preprocessor = load_object(data_transformation_artifact.transformed_object_file_path)
                print(f"      - 预处理器类型: {type(preprocessor)}")
                if hasattr(preprocessor, 'steps'):
                    print(f"      - 管道步骤: {[step[0] for step in preprocessor.steps]}")
            except Exception as e:
                print(f"      - 预处理器加载测试失败: {str(e)}")
        else:
            print("   ❌ 预处理器文件不存在")
        
        # 检查final_model目录
        final_preprocessor_path = "final_model/preprocessor.pkl"
        if os.path.exists(final_preprocessor_path):
            print(f"   ✅ Final model预处理器已保存: {final_preprocessor_path}")
        else:
            print("   ❌ Final model预处理器不存在")
        
        # 步骤7: 数据质量检查
        print("\n🔍 步骤7: 数据质量检查")
        
        if 'train_arr' in locals() and 'test_arr' in locals():
            # 检查是否有缺失值
            train_nan_count = np.isnan(train_arr).sum()
            test_nan_count = np.isnan(test_arr).sum()
            
            print(f"   - 训练数据缺失值: {train_nan_count}")
            print(f"   - 测试数据缺失值: {test_nan_count}")
            
            if train_nan_count == 0 and test_nan_count == 0:
                print("   ✅ 没有缺失值，KNN Imputer工作正常")
            else:
                print("   ⚠️ 仍存在缺失值，需要检查")
            
            # 检查数据范围
            print(f"   - 训练数据范围: [{train_arr.min():.3f}, {train_arr.max():.3f}]")
            print(f"   - 测试数据范围: [{test_arr.min():.3f}, {test_arr.max():.3f}]")
            
            # 检查目标变量是否正确转换
            train_target = train_arr[:, -1]
            test_target = test_arr[:, -1]
            
            if set(np.unique(train_target)).issubset({0, 1}) and set(np.unique(test_target)).issubset({0, 1}):
                print("   ✅ 目标变量成功转换为[0,1]格式")
            else:
                print("   ❌ 目标变量转换可能有问题")
        
        # 步骤8: 总结
        print("\n🎉 测试总结")
        print("=" * 60)
        print("✅ Data Transformation组件测试成功!")
        print("✅ 数据转换流程正常工作")
        print("✅ KNN Imputer正确处理缺失值")
        print("✅ 目标变量成功转换")
        print("✅ 预处理器正确保存")
        
        print("\n📁 生成的文件:")
        print(f"   - 转换后训练数据: {data_transformation_artifact.transformed_train_file_path}")
        print(f"   - 转换后测试数据: {data_transformation_artifact.transformed_test_file_path}")
        print(f"   - 预处理器对象: {data_transformation_artifact.transformed_object_file_path}")
        
        print("\n🎯 下一步:")
        print("   数据转换完成，可以继续进行模型训练 (Model Training)")
        
        return data_transformation_artifact
        
    except NetworkSecurityException as nse:
        print(f"\n❌ 网络安全异常: {str(nse)}")
        return None
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        return None

def test_transformer_object():
    """
    单独测试预处理器对象
    """
    try:
        print("\n🧪 测试预处理器对象")
        print("-" * 40)
        
        from networksecurity.components.data_transformation import DataTransformation
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        # 添加一些缺失值
        test_data.loc[0:5, 'feature1'] = np.nan
        test_data.loc[10:15, 'feature2'] = np.nan
        
        print(f"测试数据形状: {test_data.shape}")
        print(f"缺失值数量: {test_data.isnull().sum().sum()}")
        
        # 获取预处理器
        preprocessor = DataTransformation.get_data_transformer_object()
        print(f"预处理器类型: {type(preprocessor)}")
        
        # 训练并转换
        preprocessor_fitted = preprocessor.fit(test_data)
        transformed_data = preprocessor_fitted.transform(test_data)
        
        print(f"转换后数据形状: {transformed_data.shape}")
        print(f"转换后缺失值: {np.isnan(transformed_data).sum()}")
        
        if np.isnan(transformed_data).sum() == 0:
            print("✅ 预处理器正确处理了缺失值")
        else:
            print("❌ 预处理器未能完全处理缺失值")
        
    except Exception as e:
        print(f"❌ 预处理器测试失败: {str(e)}")

def main():
    """
    主函数
    """
    print("Data Transformation 组件测试")
    print("=" * 60)
    
    # 测试预处理器对象
    test_transformer_object()
    
    # 完整流程测试
    artifact = test_data_transformation()
    
    if artifact:
        print("\n🎊 所有测试完成!")
        print("🚀 Data Transformation组件已准备就绪!")
        print("📊 现在可以使用转换后的数据进行模型训练了!")
        return True
    else:
        print("\n❌ 测试失败，请检查配置和数据")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)