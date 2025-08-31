from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
import sys
import numpy as np

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        ClassificationMetricArtifact: 包含各项指标的对象
    """
    try:
        # 计算基本指标
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        
        return classification_metric
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def get_detailed_classification_score(y_true, y_pred, target_names=None):
    """
    获取详细的分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称，默认为None
        
    Returns:
        详细的分类指标字典
    """
    try:
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 分类报告
        class_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        # 计算真正率、假正率等
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        detailed_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'classification_report': class_report,
            'support': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true == 1)),
                'negative_samples': int(np.sum(y_true == 0))
            }
        }
        
        return detailed_metrics
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def evaluate_binary_classification(y_true, y_pred, y_pred_proba=None):
    """
    专门用于二分类评估的函数
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率（可选）
        
    Returns:
        详细的二分类评估结果
    """
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
        
        # 基本分类指标
        basic_metrics = get_detailed_classification_score(y_true, y_pred, 
                                                         target_names=['Legitimate', 'Phishing'])
        
        # 如果提供了概率，计算ROC和PR曲线相关指标
        if y_pred_proba is not None:
            # 取正类概率（假设是第二列）
            if y_pred_proba.ndim == 2:
                proba_positive = y_pred_proba[:, 1]
            else:
                proba_positive = y_pred_proba
            
            # ROC AUC
            roc_auc = roc_auc_score(y_true, proba_positive)
            
            # ROC曲线
            fpr, tpr, _ = roc_curve(y_true, proba_positive)
            
            # Precision-Recall曲线
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba_positive)
            
            # 平均精确度
            from sklearn.metrics import average_precision_score
            avg_precision = average_precision_score(y_true, proba_positive)
            
            basic_metrics.update({
                'roc_auc_score': roc_auc,
                'average_precision_score': avg_precision,
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                },
                'precision_recall_curve': {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist()
                }
            })
        
        return basic_metrics
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def print_classification_summary(metrics_dict, model_name="Model"):
    """
    打印分类结果摘要
    
    Args:
        metrics_dict: 指标字典
        model_name: 模型名称
    """
    try:
        print(f"\n{model_name} Classification Results:")
        print("=" * 50)
        print(f"Accuracy:  {metrics_dict.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics_dict.get('precision', 0):.4f}")
        print(f"Recall:    {metrics_dict.get('recall', 0):.4f}")
        print(f"F1-Score:  {metrics_dict.get('f1_score', 0):.4f}")
        
        if 'roc_auc_score' in metrics_dict:
            print(f"ROC AUC:   {metrics_dict['roc_auc_score']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = metrics_dict.get('confusion_matrix', [[0, 0], [0, 0]])
        print(f"              Predicted")
        print(f"           Legit  Phish")
        print(f"Actual Legit {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Phish {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        support = metrics_dict.get('support', {})
        if support:
            print(f"\nDataset Info:")
            print(f"Total Samples: {support.get('total_samples', 0)}")
            print(f"Positive (Phishing): {support.get('positive_samples', 0)}")
            print(f"Negative (Legitimate): {support.get('negative_samples', 0)}")
        
    except Exception as e:
        print(f"Error printing summary: {str(e)}")

def compare_models(model_results_dict):
    """
    比较多个模型的性能
    
    Args:
        model_results_dict: {model_name: metrics_dict} 格式的字典
        
    Returns:
        比较结果的DataFrame
    """
    try:
        import pandas as pd
        
        comparison_data = []
        
        for model_name, metrics in model_results_dict.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0)
            }
            
            if 'roc_auc_score' in metrics:
                row['ROC AUC'] = metrics['roc_auc_score']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 按F1分数排序
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)