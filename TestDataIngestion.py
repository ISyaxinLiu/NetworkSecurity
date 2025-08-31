#!/usr/bin/env python3
"""
测试完整的PostgreSQL Data Ingestion设置
模拟老师的使用方式
"""

import sys
import os
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_complete_setup():
    """
    完整测试PostgreSQL版本的数据摄取
    """
    try:
        print("🚀 开始测试完整的PostgreSQL Data Ingestion设置")
        print("=" * 60)
        
        # 导入修改后的模块
        from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
        from networksecurity.components.data_ingestion import DataIngestion
        from networksecurity.exception.exception import NetworkSecurityException
        from networksecurity.logging.logger import logging
        
        print("✅ 所有模块导入成功")
        
        # 步骤1: 创建训练管道配置
        print("\n📋 步骤1: 创建训练管道配置")
        training_pipeline_config = TrainingPipelineConfig()
        print(f"   - 管道名称: {training_pipeline_config.pipeline_name}")
        print(f"   - 工件目录: {training_pipeline_config.artifact_dir}")
        print(f"   - 时间戳: {training_pipeline_config.timestamp}")
        
        # 步骤2: 创建数据摄取配置
        print("\n📋 步骤2: 创建数据摄取配置")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        print(f"   - 数据库名: {data_ingestion_config.database_name}")
        print(f"   - 表名(collection_name): {data_ingestion_config.collection_name}")
        print(f"   - 特征存储路径: {data_ingestion_config.feature_store_file_path}")
        print(f"   - 训练文件路径: {data_ingestion_config.training_file_path}")
        print(f"   - 测试文件路径: {data_ingestion_config.testing_file_path}")
        print(f"   - 分割比例: {data_ingestion_config.train_test_split_ratio}")
        
        # 步骤3: 创建数据摄取对象
        print("\n📋 步骤3: 创建数据摄取对象")
        data_ingestion = DataIngestion(data_ingestion_config)
        print("✅ 数据摄取对象创建成功")
        
        # 步骤4: 检查数据库连接和数据
        print("\n📊 步骤4: 检查数据库信息")
        db_info = data_ingestion.get_database_info()
        if db_info:
            print(f"   ✅ 数据库连接成功")
            print(f"   - 数据库: {db_info['database']}")
            print(f"   - 表: {db_info['table']}")
            print(f"   - 总记录数: {db_info['total_rows']}")
            print(f"   - 钓鱼网站: {db_info['phishing_count']}")
            print(f"   - 合法网站: {db_info['legitimate_count']}")
            if 'phishing_ratio' in db_info:
                print(f"   - 钓鱼比例: {db_info['phishing_ratio']:.2%}")
        else:
            print("   ❌ 无法获取数据库信息，但将继续尝试数据摄取")
        
        # 步骤5: 执行数据摄取 (这是核心测试)
        print("\n🔄 步骤5: 执行数据摄取流程")
        print("   这将执行以下操作:")
        print("   1. 从PostgreSQL读取数据")
        print("   2. 导出到特征存储")
        print("   3. 分割训练/测试数据")
        
        # 开始数据摄取
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        print("✅ 数据摄取流程完成!")
        
        # 步骤6: 验证结果
        print("\n📁 步骤6: 验证生成的文件")
        
        # 检查训练文件
        if os.path.exists(data_ingestion_artifact.trained_file_path):
            import pandas as pd
            train_df = pd.read_csv(data_ingestion_artifact.trained_file_path)
            print(f"   ✅ 训练文件: {len(train_df)} 行, {len(train_df.columns)} 列")
            print(f"      路径: {data_ingestion_artifact.trained_file_path}")
            
            # 显示训练数据的目标变量分布
            if 'Result' in train_df.columns:
                result_dist = train_df['Result'].value_counts()
                print(f"      目标变量分布: {result_dist.to_dict()}")
        else:
            print(f"   ❌ 训练文件不存在: {data_ingestion_artifact.trained_file_path}")
        
        # 检查测试文件
        if os.path.exists(data_ingestion_artifact.test_file_path):
            test_df = pd.read_csv(data_ingestion_artifact.test_file_path)
            print(f"   ✅ 测试文件: {len(test_df)} 行, {len(test_df.columns)} 列")
            print(f"      路径: {data_ingestion_artifact.test_file_path}")
            
            # 显示测试数据的目标变量分布
            if 'Result' in test_df.columns:
                result_dist = test_df['Result'].value_counts()
                print(f"      目标变量分布: {result_dist.to_dict()}")
        else:
            print(f"   ❌ 测试文件不存在: {data_ingestion_artifact.test_file_path}")
        
        # 检查特征存储文件
        if os.path.exists(data_ingestion_config.feature_store_file_path):
            feature_df = pd.read_csv(data_ingestion_config.feature_store_file_path)
            print(f"   ✅ 特征存储文件: {len(feature_df)} 行, {len(feature_df.columns)} 列")
            print(f"      路径: {data_ingestion_config.feature_store_file_path}")
        else:
            print(f"   ❌ 特征存储文件不存在")
        
        # 步骤7: 总结
        print("\n🎉 测试总结")
        print("=" * 60)
        print("✅ PostgreSQL Data Ingestion设置测试成功!")
        print("✅ 所有组件正常工作")
        print("✅ 数据成功从PostgreSQL读取")
        print("✅ 训练/测试数据分割完成")
        print("✅ 文件生成正常")
        print("\n📊 生成的文件:")
        print(f"   - 特征存储: {data_ingestion_config.feature_store_file_path}")
        print(f"   - 训练数据: {data_ingestion_artifact.trained_file_path}")
        print(f"   - 测试数据: {data_ingestion_artifact.test_file_path}")
        
        print("\n🎯 下一步:")
        print("   现在你可以将这些文件用于下一个阶段的数据验证和模型训练!")
        
        return data_ingestion_artifact
        
    except NetworkSecurityException as nse:
        print(f"\n❌ 网络安全异常: {str(nse)}")
        print("请检查:")
        print("1. PostgreSQL服务是否运行")
        print("2. 数据库连接配置是否正确")
        print("3. 表 'NetworkData' 是否存在且有数据")
        return None
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print("请检查:")
        print("1. 所有依赖是否已安装 (pip install psycopg2-binary sqlalchemy)")
        print("2. .env 文件配置是否正确")
        print("3. 项目路径和导入是否正确")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        return None

def quick_environment_check():
    """
    快速环境检查
    """
    print("🔍 环境检查")
    print("-" * 30)
    
    # 检查Python包
    try:
        import psycopg2
        print("✅ psycopg2 已安装")
    except ImportError:
        print("❌ psycopg2 未安装，请运行: pip install psycopg2-binary")
        return False
    
    try:
        import sqlalchemy
        print("✅ sqlalchemy 已安装")
    except ImportError:
        print("❌ sqlalchemy 未安装，请运行: pip install sqlalchemy")
        return False
    
    try:
        import pandas
        print("✅ pandas 已安装")
    except ImportError:
        print("❌ pandas 未安装，请运行: pip install pandas")
        return False
    
    try:
        import sklearn
        print("✅ sklearn 已安装")
    except ImportError:
        print("❌ sklearn 未安装，请运行: pip install scikit-learn")
        return False
    
    # 检查环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    required_env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            # 不显示密码的完整值
            display_value = "***" if var == 'DB_PASSWORD' else value
            print(f"✅ {var}={display_value}")
        else:
            print(f"❌ {var} 未设置")
            return False
    
    print("✅ 环境检查通过")
    return True

def test_database_connection():
    """
    测试数据库连接
    """
    print("\n🔗 测试数据库连接")
    print("-" * 30)
    
    try:
        from dotenv import load_dotenv
        import psycopg2
        
        load_dotenv()
        
        # 获取连接参数
        conn_params = {
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
            'database': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD")
        }
        
        print(f"尝试连接到: {conn_params['user']}@{conn_params['host']}:{conn_params['port']}/{conn_params['database']}")
        
        # 测试连接
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        print(f"✅ 成功连接到数据库: {conn_params['database']}")
        
        # 检查表是否存在
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'NetworkData'
        """)
        
        tables = cursor.fetchall()
        if tables:
            print("✅ 表 'NetworkData' 存在")
            
            # 检查记录数
            cursor.execute('SELECT COUNT(*) FROM "NetworkData"')
            count = cursor.fetchone()[0]
            print(f"✅ 表中有 {count} 条记录")
            
            if count > 0:
                # 检查结果分布
                cursor.execute('SELECT "Result", COUNT(*) FROM "NetworkData" GROUP BY "Result"')
                result_dist = cursor.fetchall()
                print(f"✅ 数据分布: {dict(result_dist)}")
                
                # 检查列名
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'NetworkData' AND table_schema = 'public'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cursor.fetchall()]
                print(f"✅ 表结构: {len(columns)} 列")
                print(f"   前10列: {columns[:10]}")
                
            else:
                print("⚠️ 表中没有数据，请先运行数据导入")
                return False
        else:
            print("❌ 表 'NetworkData' 不存在")
            print("请确保已经运行了数据导入脚本")
            print("可用的表:")
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            available_tables = cursor.fetchall()
            for table in available_tables:
                print(f"   - {table[0]}")
            return False
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {str(e)}")
        print("请检查:")
        print("1. PostgreSQL服务是否启动")
        print("2. 数据库连接参数是否正确")
        print("3. 用户是否有访问权限")
        print("4. 防火墙设置是否正确")
        return False

def test_project_structure():
    """
    检查项目结构
    """
    print("\n📁 检查项目结构")
    print("-" * 30)
    
    required_files = [
        'networksecurity/__init__.py',
        'networksecurity/constant/__init__.py',
        'networksecurity/constant/training_pipeline/__init__.py',
        'networksecurity/entity/__init__.py',
        'networksecurity/entity/config_entity.py',
        'networksecurity/entity/artifact_entity.py',
        'networksecurity/components/__init__.py',
        'networksecurity/components/data_ingestion.py',
        'networksecurity/exception/__init__.py',
        'networksecurity/exception/exception.py',
        'networksecurity/logging/__init__.py',
        'networksecurity/logging/logger.py',
        '.env'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺少 {len(missing_files)} 个文件")
        return False
    
    print("✅ 项目结构完整")
    return True

def main():
    """
    主函数
    """
    print("PostgreSQL Data Ingestion 完整测试")
    print("=" * 60)
    
    # 步骤1: 检查项目结构
    print("步骤1: 检查项目结构")
    if not test_project_structure():
        print("\n❌ 项目结构不完整，请确保所有必要文件都存在")
        return False
    
    # 步骤2: 环境检查
    print("\n步骤2: 环境检查")
    if not quick_environment_check():
        print("\n❌ 环境检查失败，请先安装必要的依赖")
        return False
    
    # 步骤3: 数据库连接测试
    print("\n步骤3: 数据库连接测试")
    if not test_database_connection():
        print("\n❌ 数据库连接测试失败，请检查数据库配置")
        return False
    
    # 步骤4: 完整功能测试
    print("\n步骤4: 完整功能测试")
    print("=" * 60)
    artifact = test_complete_setup()
    
    if artifact:
        print("\n🎊 恭喜! 所有测试通过")
        print("🚀 你的PostgreSQL Data Ingestion组件已经可以正常工作!")
        print("📝 现在可以继续进行数据验证和模型训练步骤了")
        print("\n📋 下一步建议:")
        print("1. 运行数据验证 (Data Validation)")
        print("2. 进行数据转换 (Data Transformation)")
        print("3. 开始模型训练 (Model Training)")
        return True
    else:
        print("\n❌ 测试失败，请根据上面的错误信息进行修复")
        print("\n🔧 常见解决方案:")
        print("1. 检查 .env 文件配置")
        print("2. 确保PostgreSQL服务运行")
        print("3. 验证数据库表和数据存在")
        print("4. 检查Python包依赖")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)