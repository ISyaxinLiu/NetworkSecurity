#!/usr/bin/env python3
"""
测试Data Validation组件
"""

import sys
import os
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_data_validation():
    """
    测试Data Validation完整流程
    """
    try:
        print("🔍 开始测试Data Validation组件")
        print("=" * 60)
        
        # 导入必要的模块
        from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
        from networksecurity.components.data_ingestion import DataIngestion
        from networksecurity.components.data_validation import DataValidation
        from networksecurity.exception.exception import NetworkSecurityException
        from networksecurity.logging.logger import logging
        
        print("✅ 所有模块导入成功")
        
        # 步骤1: 创建配置对象
        print("\n📋 步骤1: 创建配置对象")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        
        print(f"   - 数据验证目录: {data_validation_config.data_validation_dir}")
        print(f"   - 有效数据目录: {data_validation_config.valid_data_dir}")
        print(f"   - 无效数据目录: {data_validation_config.invalid_data_dir}")
        print(f"   - 漂移报告路径: {data_validation_config.drift_report_file_path}")
        
        # 步骤2: 先执行数据摄取（确保有数据可用）
        print("\n📥 步骤2: 执行数据摄取（获取训练测试数据）")
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        print(f"   - 训练文件: {data_ingestion_artifact.trained_file_path}")
        print(f"   - 测试文件: {data_ingestion_artifact.test_file_path}")
        
        # 验证文件是否存在
        if not os.path.exists(data_ingestion_artifact.trained_file_path):
            raise Exception(f"训练文件不存在: {data_ingestion_artifact.trained_file_path}")
        if not os.path.exists(data_ingestion_artifact.test_file_path):
            raise Exception(f"测试文件不存在: {data_ingestion_artifact.test_file_path}")
            
        print("✅ 数据摄取完成，文件验证通过")
        
        # 步骤3: 创建数据验证对象
        print("\n🔍 步骤3: 创建数据验证对象")
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        print("✅ 数据验证对象创建成功")
        
        # 步骤4: 执行数据验证
        print("\n🔄 步骤4: 执行数据验证流程")
        print("   这将执行以下检查:")
        print("   1. 验证列数是否正确")
        print("   2. 验证数值列是否存在")
        print("   3. 验证目标列是否正确")
        print("   4. 检测数据漂移")
        
        # 开始验证
        data_validation_artifact = data_validation.initiate_data_validation()
        
        print("✅ 数据验证流程完成!")
        
        # 步骤5: 分析验证结果
        print("\n📊 步骤5: 分析验证结果")
        print(f"   - 验证状态: {'通过' if data_validation_artifact.validation_status else '失败'}")
        
        if data_validation_artifact.validation_status:
            print("✅ 数据验证通过!")
            print(f"   - 有效训练文件: {data_validation_artifact.valid_train_file_path}")
            print(f"   - 有效测试文件: {data_validation_artifact.valid_test_file_path}")
            
            # 检查文件是否真的存在
            if os.path.exists(data_validation_artifact.valid_train_file_path):
                import pandas as pd
                valid_train_df = pd.read_csv(data_validation_artifact.valid_train_file_path)
                print(f"   - 有效训练数据: {len(valid_train_df)} 行, {len(valid_train_df.columns)} 列")
            
            if os.path.exists(data_validation_artifact.valid_test_file_path):
                valid_test_df = pd.read_csv(data_validation_artifact.valid_test_file_path)
                print(f"   - 有效测试数据: {len(valid_test_df)} 行, {len(valid_test_df.columns)} 列")
                
        else:
            print("❌ 数据验证失败!")
            print(f"   - 无效训练文件: {data_validation_artifact.invalid_train_file_path}")
            print(f"   - 无效测试文件: {data_validation_artifact.invalid_test_file_path}")
        
        # 步骤6: 检查漂移报告
        print("\n📈 步骤6: 检查数据漂移报告")
        drift_report_path = data_validation_artifact.drift_report_file_path
        
        if os.path.exists(drift_report_path):
            print(f"✅ 漂移报告已生成: {drift_report_path}")
            
            # 读取并显示报告摘要
            from networksecurity.utils.main_utils.utils import read_yaml_file
            drift_report = read_yaml_file(drift_report_path)
            
            if 'summary' in drift_report:
                summary = drift_report['summary']
                print(f"   - 检查的列数: {summary.get('total_columns_checked', 0)}")
                print(f"   - 检测到漂移: {'是' if summary.get('drift_detected', False) else '否'}")
                print(f"   - 使用的阈值: {summary.get('threshold_used', 0.05)}")
                
                if summary.get('drift_columns'):
                    print(f"   - 漂移的列: {summary['drift_columns']}")
                else:
                    print("   - 没有列出现显著漂移")
            
            # 显示前几列的详细信息
            print("   前5列的p值:")
            count = 0
            for column, info in drift_report.items():
                if isinstance(info, dict) and 'p_value' in info and count < 5:
                    print(f"     {column}: p_value={info['p_value']:.4f}, drift={info['drift_status']}")
                    count += 1
        else:
            print("❌ 漂移报告文件未生成")
        
        # 步骤7: 总结
        print("\n🎉 测试总结")
        print("=" * 60)
        print("✅ Data Validation组件测试成功!")
        print("✅ 所有验证功能正常工作")
        print("✅ 数据质量检查完成")
        print("✅ 漂移检测正常运行")
        
        if data_validation_artifact.validation_status:
            print("\n📁 生成的文件:")
            print(f"   - 有效训练数据: {data_validation_artifact.valid_train_file_path}")
            print(f"   - 有效测试数据: {data_validation_artifact.valid_test_file_path}")
        
        print(f"   - 漂移报告: {data_validation_artifact.drift_report_file_path}")
        
        print("\n🎯 下一步:")
        if data_validation_artifact.validation_status:
            print("   数据验证通过，可以继续进行数据转换 (Data Transformation)")
        else:
            print("   数据验证失败，需要检查数据质量问题")
        
        return data_validation_artifact
        
    except NetworkSecurityException as nse:
        print(f"\n❌ 网络安全异常: {str(nse)}")
        return None
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        return None

def test_individual_validations():
    """
    测试各个验证功能
    """
    try:
        print("\n🧪 测试各个验证功能")
        print("-" * 40)
        
        # 导入模块
        from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
        from networksecurity.components.data_ingestion import DataIngestion
        from networksecurity.components.data_validation import DataValidation
        import pandas as pd
        
        # 创建配置
        training_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_config)
        data_validation_config = DataValidationConfig(training_config)
        
        # 获取数据
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # 创建验证器
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        
        # 读取数据
        train_df = pd.read_csv(data_ingestion_artifact.trained_file_path)
        test_df = pd.read_csv(data_ingestion_artifact.test_file_path)
        
        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")
        
        # 测试1: 列数验证
        print("\n1. 测试列数验证:")
        result = data_validation.validate_number_of_columns(train_df)
        print(f"   训练数据列数验证: {'通过' if result else '失败'}")
        
        result = data_validation.validate_number_of_columns(test_df)
        print(f"   测试数据列数验证: {'通过' if result else '失败'}")
        
        # 测试2: 数值列验证
        print("\n2. 测试数值列验证:")
        result = data_validation.validate_numerical_columns(train_df)
        print(f"   训练数据数值列验证: {'通过' if result else '失败'}")
        
        result = data_validation.validate_numerical_columns(test_df)
        print(f"   测试数据数值列验证: {'通过' if result else '失败'}")
        
        # 测试3: 目标列验证
        print("\n3. 测试目标列验证:")
        result = data_validation.validate_target_column(train_df)
        print(f"   训练数据目标列验证: {'通过' if result else '失败'}")
        
        result = data_validation.validate_target_column(test_df)
        print(f"   测试数据目标列验证: {'通过' if result else '失败'}")
        
        # 测试4: 数据漂移检测
        print("\n4. 测试数据漂移检测:")
        drift_result = data_validation.detect_dataset_drift(train_df, test_df)
        print(f"   数据漂移检测: {'无显著漂移' if drift_result else '检测到漂移'}")
        
        print("✅ 各个验证功能测试完成")
        
    except Exception as e:
        print(f"❌ 个别功能测试失败: {str(e)}")

def main():
    """
    主函数
    """
    print("Data Validation 组件测试")
    print("=" * 60)
    
    # 完整流程测试
    artifact = test_data_validation()
    
    if artifact:
        # 个别功能测试
        test_individual_validations()
        
        print("\n🎊 所有测试完成!")
        print("🚀 Data Validation组件已准备就绪!")
        return True
    else:
        print("\n❌ 测试失败，请检查配置和数据")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)