import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 新增导入：Optuna和并行化
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, falling back to GridSearchCV")

from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)

def evaluate_classification_models(X_train, y_train, X_test, y_test, models, param):
    """
    专门用于分类任务的模型评估函数 - 原版GridSearch
    适合网络安全钓鱼检测项目
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]
            
            # 网格搜索
            gs = GridSearchCV(model, para, cv=3, scoring='f1')  # 改用f1评分
            gs.fit(X_train, y_train)
            
            # 训练最佳模型
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # 预测
            y_test_pred = model.predict(X_test)
            
            # 计算分类指标
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            report[model_name] = {
                'accuracy': accuracy,
                'precision_score': precision,  # 保持与原代码一致的key名
                'recall_score': recall,
                'f1_score': f1,
                'best_params': gs.best_params_
            }
        
        return report
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

# ================== 新增：Optuna优化版本 ==================

def _optimize_single_model_optuna(model_name, model, param_space, X_train, y_train, X_test, y_test, n_trials=50):
    """
    单个模型的Optuna优化 - 用于并行调用
    """
    try:
        def objective(trial):
            # 根据参数空间生成参数
            params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, list):
                    # 离散参数
                    if all(isinstance(x, (int, float)) for x in param_config):
                        if all(isinstance(x, int) for x in param_config):
                            params[param_name] = trial.suggest_categorical(param_name, param_config)
                        else:
                            params[param_name] = trial.suggest_categorical(param_name, param_config)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_config)
            
            # 创建模型
            from copy import deepcopy
            temp_model = deepcopy(model)
            temp_model.set_params(**params)
            
            # 训练和评估
            temp_model.fit(X_train, y_train)
            y_pred = temp_model.predict(X_test)
            
            # 返回F1分数
            f1 = f1_score(y_test, y_pred, average='weighted')
            return f1
        
        # 创建study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10)
        )
        
        # 优化
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 用最佳参数重新训练
        best_model = model.__class__(**study.best_params)
        best_model.fit(X_train, y_train)
        
        # 计算最终指标
        y_pred = best_model.predict(X_test)
        
        result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_score': precision_score(y_test, y_pred, average='weighted'),
            'recall_score': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'best_value': study.best_value
        }
        
        logging.info(f"{model_name} Optuna优化完成 - F1: {result['f1_score']:.4f}, 最佳参数: {study.best_params}")
        return model_name, result
        
    except Exception as e:
        logging.error(f"{model_name} Optuna优化失败: {str(e)}")
        # 降级到GridSearch
        return _optimize_single_model_gridsearch(model_name, model, param_space, X_train, y_train, X_test, y_test)

def _optimize_single_model_gridsearch(model_name, model, param_space, X_train, y_train, X_test, y_test):
    """
    单个模型的GridSearch优化 - 备用方案
    """
    try:
        # 转换参数格式
        if isinstance(param_space, dict):
            param_grid = param_space
        else:
            param_grid = param_space
            
        gs = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted')
        gs.fit(X_train, y_train)
        
        y_pred = gs.predict(X_test)
        
        result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_score': precision_score(y_test, y_pred, average='weighted'),
            'recall_score': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'best_params': gs.best_params_
        }
        
        logging.info(f"{model_name} GridSearch完成 - F1: {result['f1_score']:.4f}")
        return model_name, result
        
    except Exception as e:
        logging.error(f"{model_name} GridSearch失败: {str(e)}")
        raise e

def evaluate_classification_models_parallel(X_train, y_train, X_test, y_test, models, param, 
                                           use_optuna=True, n_trials=50, n_jobs=-1):
    """
    🔥 新增：并行化 + Optuna优化的分类模型评估
    
    Args:
        X_train, y_train, X_test, y_test: 训练测试数据
        models: 模型字典
        param: 参数字典
        use_optuna: 是否使用Optuna（默认True）
        n_trials: Optuna试验次数
        n_jobs: 并行进程数（-1使用所有CPU）
    
    Returns:
        dict: 模型评估报告
    """
    try:
        logging.info(f"开始并行模型评估 - 使用Optuna: {use_optuna and OPTUNA_AVAILABLE}, 并行度: {n_jobs}")
        
        if n_jobs == -1:
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), len(models))
        
        # 选择优化函数
        if use_optuna and OPTUNA_AVAILABLE:
            optimize_func = _optimize_single_model_optuna
            extra_args = (n_trials,)
            logging.info(f"使用Optuna贝叶斯优化，每个模型{n_trials}次试验")
        else:
            optimize_func = _optimize_single_model_gridsearch
            extra_args = ()
            logging.info("使用GridSearch网格搜索")
        
        # 并行优化所有模型
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(optimize_func)(
                model_name, model, param[model_name], 
                X_train, y_train, X_test, y_test, *extra_args
            ) 
            for model_name, model in models.items()
        )
        
        # 整理结果
        report = {}
        for model_name, result in results:
            report[model_name] = result
        
        # 打印排序结果
        sorted_models = sorted(report.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        logging.info("=== 模型性能排序 (按F1分数) ===")
        for i, (name, metrics) in enumerate(sorted_models, 1):
            logging.info(f"{i}. {name}: F1={metrics['f1_score']:.4f}, "
                        f"Precision={metrics['precision_score']:.4f}, "
                        f"Recall={metrics['recall_score']:.4f}")
        
        return report
        
    except Exception as e:
        logging.error(f"并行模型评估失败，降级到串行GridSearch: {str(e)}")
        return evaluate_classification_models(X_train, y_train, X_test, y_test, models, param)