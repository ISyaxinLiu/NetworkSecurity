#!/usr/bin/env python3
"""
MLflow修复和测试工具
"""

import os
import sys
import mlflow
import requests
from datetime import datetime
import dagshub

def setup_mlflow_config():
    """
    设置正确的MLflow配置
    """
    # 你的正确配置
    mlflow_config = {
        "MLFLOW_TRACKING_URI": "https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow",
        "MLFLOW_TRACKING_USERNAME": "ISyaxinLiu",
        "MLFLOW_TRACKING_PASSWORD": "f36ccefd8aa76fa3d07a50e0baf446776f28f379"
    }
    
    # 设置环境变量
    for key, value in mlflow_config.items():
        os.environ[key] = value
    
    print("MLflow配置已设置:")
    print(f"  URI: {mlflow_config['MLFLOW_TRACKING_URI']}")
    print(f"  用户名: {mlflow_config['MLFLOW_TRACKING_USERNAME']}")
    print(f"  Token: {mlflow_config['MLFLOW_TRACKING_PASSWORD'][:8]}...")
    
    return mlflow_config

def test_dagshub_connection():
    """
    测试DagHub连接
    """
    print("\n测试DagHub连接...")
    
    try:
        # 初始化DagHub
        dagshub.init(repo_owner='ISyaxinLiu', repo_name='NetworkSecurity', mlflow=True)
        print("DagHub初始化成功")
        
        # 测试API访问
        api_url = "https://dagshub.com/api/v1/repos/ISyaxinLiu/NetworkSecurity"
        headers = {
            "Authorization": f"token {os.environ['MLFLOW_TRACKING_PASSWORD']}"
        }
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("DagHub API访问成功")
            return True
        else:
            print(f"DagHub API访问失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"DagHub连接测试失败: {str(e)}")
        return False

def test_mlflow_experiment():
    """
    测试创建MLflow实验
    """
    print("\n测试MLflow实验创建...")
    
    try:
        # 设置registry URI
        mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
        
        # 获取或创建实验
        experiment_name = "NetworkSecurity_Test"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"创建新实验: {experiment_name} (ID: {experiment_id})")
        except Exception:
            # 实验可能已存在
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                print(f"使用现有实验: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = "0"  # 使用默认实验
                print("使用默认实验 (ID: 0)")
        
        # 设置实验
        mlflow.set_experiment(experiment_name if experiment_name else "Default")
        
        # 创建运行
        with mlflow.start_run(run_name=f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # 记录测试数据
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("f1_score", 0.93)
            mlflow.log_metric("precision", 0.94)
            mlflow.log_metric("recall", 0.92)
            
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("test_run", True)
            
            # 记录标签
            mlflow.set_tag("project", "NetworkSecurity")
            mlflow.set_tag("environment", "test")
            
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            
            print(f"测试运行创建成功:")
            print(f"  运行ID: {run_id}")
            print(f"  实验ID: {experiment_id}")
            print(f"  查看链接: https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow/#/experiments/{experiment_id}/runs/{run_id}")
        
        return True
        
    except Exception as e:
        print(f"MLflow实验测试失败: {str(e)}")
        return False

def create_fixed_model_trainer_snippet():
    """
    创建修复后的model_trainer代码片段
    """
    print("\n修复后的track_mlflow函数:")
    print("=" * 50)
    
    fixed_code = '''
def track_mlflow(self, best_model, classificationmetric, model_name: str):
    """
    修复后的MLflow跟踪函数
    """
    try:
        # 使用你的正确配置
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "ISyaxinLiu" 
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "f36ccefd8aa76fa3d07a50e0baf446776f28f379"
        
        # 初始化DagHub
        import dagshub
        dagshub.init(repo_owner='ISyaxinLiu', repo_name='NetworkSecurity', mlflow=True)
        
        mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
        
        # 创建或获取实验
        experiment_name = "NetworkSecurity_Phishing_Detection"
        try:
            mlflow.set_experiment(experiment_name)
        except:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        
        # 开始运行
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # 记录指标
            mlflow.log_metric("f1_score", float(classificationmetric.f1_score))
            mlflow.log_metric("precision", float(classificationmetric.precision_score))
            mlflow.log_metric("recall_score", float(classificationmetric.recall_score))
            
            # 记录参数
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("data_source", "postgresql")
            mlflow.log_param("pipeline_version", "1.0")
            
            # 记录标签
            mlflow.set_tag("project", "NetworkSecurity")
            mlflow.set_tag("task", "binary_classification")
            mlflow.set_tag("dataset", "phishing_detection")
            
            # 记录模型
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    best_model, 
                    "model", 
                    registered_model_name=f"NetworkSecurity_{model_name}"
                )
            else:
                mlflow.sklearn.log_model(best_model, "model")
            
            print(f"MLflow运行完成: {run_name}")
            
    except Exception as e:
        logging.error(f"MLflow跟踪失败: {str(e)}")
        # 确保运行状态正确结束
        try:
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
        except:
            pass
    '''
    
    print(fixed_code)

def main():
    """
    主函数
    """
    print("MLflow问题诊断和修复")
    print("=" * 60)
    
    # 设置配置
    setup_mlflow_config()
    
    # 测试连接
    dagshub_ok = test_dagshub_connection()
    mlflow_ok = test_mlflow_experiment()
    
    if dagshub_ok and mlflow_ok:
        print("\n所有测试通过!")
        print("现在应该可以看到MLflow实验了")
        print("请检查: https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow")
    else:
        print("\n发现问题，请按以下步骤修复:")
        create_fixed_model_trainer_snippet()
        
        print("\n快速解决方案:")
        print("1. 检查网络连接")
        print("2. 确认DagHub仓库访问权限")  
        print("3. 尝试重新运行模型训练")
        print("4. 或者暂时禁用MLflow专注于核心功能")

if __name__ == "__main__":
    main()