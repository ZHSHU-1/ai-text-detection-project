'''简单的数据清洗，暂时用中文注释和提示以便于阅读，路径与文件名均为测试用'''
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR="D:\\AI_text_detect_Project\\"

def main():
    # ---------------------- 配置部分 ----------------------
    # 清洗后数据的路径
    processed_data_dir = os.path.join(BASE_DIR,"data","processed")
    # 特征和标签的保存路径
    features_save_path = os.path.join(processed_data_dir,"features_labels_test.pkl")
    # TF-IDF模型的临时保存路径
    models_dir=os.path.join(BASE_DIR,"models")
    tfidf_temp_path = os.path.join(models_dir,"tfidf_temp_test.pkl")
    
    # 创建保存文件夹
    os.makedirs(models_dir,exist_ok=True)
    
    # ---------------------- 执行部分 ----------------------
    print("正在读取清洗后的数据...")
    processed_data_path=os.path.join(processed_data_dir,"cleaned_data_test.csv")
    df = pd.read_csv(processed_data_path)
    X_text = df['cleaned_text']  # 清洗后的文本
    y = df['label']  # 标签（0=人工，1=AI）
    
    print("正在进行TF-IDF特征提取...")
    # 初始化TF-IDF向量器：保留3000个最重要的词（最基础版本）
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    # 训练并转换文本
    X_tfidf = tfidf.fit_transform(X_text)
    
    print(f"TF-IDF特征提取完成！特征数量：{X_tfidf.shape[1]}")
    
    # ---------------------- 保存部分 ----------------------
    # 保存特征和标签（直接保存TF-IDF稀疏矩阵，不用toarray）
    joblib.dump((X_tfidf, y), features_save_path)
    # 临时保存TF-IDF模型
    joblib.dump(tfidf, tfidf_temp_path)
    
    print(f"特征提取完成！特征和标签已保存至：{features_save_path}")

if __name__ == "__main__":
    main()
