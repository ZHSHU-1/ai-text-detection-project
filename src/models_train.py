'''简单的特征提取，暂时用中文注释和提示以便于阅读，路径与文件名均为测试用,注：测试文本量过少会导致报错'''
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

BASE_DIR="D:\\AI_text_detect_Project\\"

def main():
    # ---------------------- 配置部分 ----------------------
    # 特征和标签的路径
    processed_data_dir=os.path.join(BASE_DIR,"data","processed")
    features_path=os.path.join(processed_data_dir,"features_labels_test.pkl")
    # 临时模型路径
    models_dir=os.path.join(BASE_DIR,"models")
    tfidf_temp_path = os.path.join(models_dir,"tfidf_temp_test.pkl")
    # 最终模型保存路径
    final_model_path = os.path.join(models_dir,"logistic_regression_model_test.pkl")
    final_tfidf_path = os.path.join(models_dir,"tfidf_test.pkl")
    
    # ---------------------- 执行部分 ----------------------
    print("正在加载特征和标签...")
    X_tfidf, y = joblib.load(features_path)
    
    print("正在划分训练集和测试集...")
    # 按9:1划分训练集和测试集，固定随机种子保证可复现
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.1, random_state=42, stratify=y
    )
    
    print("正在定义逻辑回归模型（单个基础模型）...")
    # 只用单个逻辑回归模型：最基础、最稳定
    model = LogisticRegression(
        class_weight="balanced",  # 适配不平衡数据（AI文本少、人工文本多）
        max_iter=1000,            # 防止模型不收敛
        random_state=42            # 固定随机种子，保证结果可复现
    )
    
    print("正在训练模型...")
    model.fit(X_train, y_train)
    print("模型训练完成！")
    
    # ---------------------- 保存部分 ----------------------
    print("正在保存最终模型...")
    # 保存逻辑回归模型
    joblib.dump(model, final_model_path)
    # 移动临时模型到最终路径
    joblib.dump(joblib.load(tfidf_temp_path), final_tfidf_path)
    # 删除临时文件
    os.remove(tfidf_temp_path)
    
    # 保存测试集（用于后续评估）
    joblib.dump((X_test, y_test), os.path.join(processed_data_dir,"test_data_test.pkl"))
    
    print(f"所有模型已保存至 models/ 文件夹！")
    print(f"测试集已保存至 data/processed/test_data.pkl")

if __name__ == "__main__":
    main()
