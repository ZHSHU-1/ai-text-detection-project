import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import os

BASE_DIR="D:\\AI_text_detect_Project\\"

def main():
    # ---------------------- 配置部分 ----------------------
    # 模型和测试集路径
    models_dir=os.path.join(BASE_DIR,"models")
    model_path = os.path.join(models_dir,"logistic_regression_model_test.pkl")
    
    processed_data_dir=os.path.join(BASE_DIR,"data","processed")
    test_data_path = os.path.join(processed_data_dir,"test_data_test.pkl")
    # 评估结果保存路径
    result_dir=os.path.join(BASE_DIR,"result")
    result_save_path = os.path.join(result_dir,"evaluation_report_test.txt")
    confusion_matrix_path = os.path.join(result_dir,"confusion_matrix_test.png")
    
    # 创建保存文件夹
    os.makedirs(result_dir, exist_ok=True)
    
    # ---------------------- 执行部分 ----------------------
    print("正在加载模型和测试集...")
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)
    
    print("正在进行预测...")
    y_pred = model.predict(X_test)
    
    print("正在计算评估指标...")
    # 计算核心指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 打印指标
    print("="*50)
    print("模型评估结果：")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print("="*50)
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred))
    
    # ---------------------- 保存部分 ----------------------
    # 保存文本报告
    with open(result_save_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("模型评估结果\n")
        f.write("="*50 + "\n")
        f.write(f"准确率 (Accuracy): {acc:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"F1分数 (F1-Score): {f1:.4f}\n")
        f.write("="*50 + "\n")
        f.write("\n详细分类报告：\n")
        f.write(classification_report(y_test, y_pred))
    
    # 绘制并保存混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Human", "AI"], 
            yticklabels=["Human", "AI"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n评估报告已保存至：{result_save_path}")
    print(f"混淆矩阵已保存至：{confusion_matrix_path}")

if __name__ == "__main__":
    main()
