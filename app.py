import sys
import os
from dotenv import load_dotenv
load_dotenv()

# PostgreSQL 配置替代 MongoDB
import psycopg2
from sqlalchemy import create_engine
import pandas as pd

# PostgreSQL 连接配置
postgresql_url = os.getenv("POSTGRESQL_URL")  # 格式: postgresql://username:password@localhost:5432/database_name
# 或者分别配置
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432") 
db_name = os.getenv("DB_NAME", "ns")
db_user = os.getenv("DB_USER", "nsproject")
db_password = os.getenv("DB_PASSWORD", "123")



# 创建PostgreSQL连接
if postgresql_url:
    engine = create_engine(postgresql_url)
else:
    engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # 读取上传的CSV文件
        df = pd.read_csv(file.file)
        
        # 加载预处理器和模型
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        
        # 添加预测结果列
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        
        # 确保输出目录存在
        os.makedirs('prediction_output', exist_ok=True)
        
        # 保存预测结果到CSV
        df.to_csv('prediction_output/output.csv', index=False)
        
        # 可选：保存到PostgreSQL数据库
        # df.to_sql('predictions', engine, if_exists='append', index=False)
        
        # 生成HTML表格
        table_html = df.to_html(classes='table table-striped')
        
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

# 新增：数据库健康检查端点
@app.get("/health/db")
async def db_health_check():
    try:
        # 测试数据库连接
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)