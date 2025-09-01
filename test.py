# 创建文件 test_mlflow_simple.py
import os
import mlflow
import dagshub

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ISyaxinLiu/NetworkSecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "ISyaxinLiu"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f36ccefd8aa76fa3d07a50e0baf446776f28f379"

dagshub.init(repo_owner='ISyaxinLiu', repo_name='NetworkSecurity', mlflow=True)

with mlflow.start_run(run_name="simple_test"):
    mlflow.log_metric("test_f1", 0.97)
    mlflow.log_param("test_model", "GradientBoosting")
    print("简单测试完成")