from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model  # 修复拼写错误：mode这个l -> model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def predict(self, x):
        """
        使用预处理器和模型进行预测
        
        Args:
            x: 输入特征数据
            
        Returns:
            预测结果
        """
        try:
            # 使用预处理器转换输入数据
            x_transform = self.preprocessor.transform(x)
            
            # 使用模型进行预测
            y_hat = self.model.predict(x_transform)
            
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def predict_proba(self, x):
        """
        预测概率（如果模型支持）
        
        Args:
            x: 输入特征数据
            
        Returns:
            预测概率
        """
        try:
            # 使用预处理器转换输入数据
            x_transform = self.preprocessor.transform(x)
            
            # 检查模型是否支持概率预测
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(x_transform)
                return y_prob
            else:
                logging.warning("Model does not support probability prediction")
                return None
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def save_model(self, file_path: str = None):
        try:
            if file_path is None:
                # 使用默认路径
                os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
                file_path = os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME)
            else:
                # 确保目录存在
                dir_path = os.path.dirname(file_path)
                if dir_path and dir_path != '':  # 检查路径不为空
                    os.makedirs(dir_path, exist_ok=True)
            
            # 保存整个NetworkModel对象
            from networksecurity.utils.main_utils.utils import save_object
            save_object(file_path, self)
            
            logging.info(f"Model saved to: {file_path}")
            return file_path
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def load_model(file_path: str):
        """
        从文件加载模型
        
        Args:
            file_path: 模型文件路径
            
        Returns:
            NetworkModel对象
        """
        try:
            from networksecurity.utils.main_utils.utils import load_object
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
            
            model = load_object(file_path)
            logging.info(f"Model loaded from: {file_path}")
            
            return model
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        try:
            info = {
                'preprocessor_type': type(self.preprocessor).__name__,
                'model_type': type(self.model).__name__,
                'has_predict_proba': hasattr(self.model, 'predict_proba')
            }
            
            # 如果模型有feature_importances_属性，添加特征重要性信息
            if hasattr(self.model, 'feature_importances_'):
                info['has_feature_importance'] = True
                info['n_features'] = len(self.model.feature_importances_)
            else:
                info['has_feature_importance'] = False
            
            return info
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)