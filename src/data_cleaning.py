'''实现简单的文本清洗，暂时使用中文保存方便阅读，路径与文件名均为测试使用'''
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

BASE_DIR="D:\\AI_text_detect_Project\\"
# 下载英文停用词（第一次运行需要，之后可以注释掉）
#nltk.download('stopwords')

def clean_text(text):
    """
    文本清洗函数：转小写、去标点、去数字、去停用词、0代表人类、1代表AI
    """
    # 1. 转小写
    text = text.lower()
    # 2. 去掉非字母字符（保留空格）
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. 分词
    words = text.split()
    # 4. 去停用词
    #stop_words = set(stopwords.words('english'))
    #words = [w for w in words if w not in stop_words]
    # 5. 重新拼接成文本
    return ' '.join(words)

def main():
    # ---------------------- 配置部分 ----------------------
    # 替换为你的原始数据集路径（CSV格式，必须包含text和label两列）
    raw_data_dir=os.path.join(BASE_DIR,"data","raw")
    raw_data_path = os.path.join(raw_data_dir,"AI_project_test_text.csv")
    # 清洗后数据的保存路径
    processed_data_dir=os.path.join(BASE_DIR,"data","processed")
    processed_data_path=os.path.join(processed_data_dir,"cleaned_data_test.csv")
    
    # 创建保存文件夹（如果不存在）
    os.makedirs(raw_data_dir,exist_ok=True)
    os.makedirs(processed_data_dir,exist_ok=True)
    
    # ---------------------- 执行部分 ----------------------
    print("正在读取原始数据...")
    df = pd.read_csv(raw_data_path)
    
    # 检查数据是否有缺失值
    print(f"原始数据缺失值情况：\n{df.isnull().sum()}")
    df = df.dropna(subset=['text', 'label'])  # 删除text或label为空的行
    
    print("正在进行文本清洗...")
    # 对text列应用清洗函数
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 只保留需要的列：清洗后的文本和标签
    df_clean = df[['cleaned_text', 'label']]
    
    # 保存清洗后的数据
    df_clean.to_csv(processed_data_path, index=False, encoding='utf-8')
    print(f"文本清洗完成！清洗后数据已保存至：{processed_data_path}")
    print(f"清洗后数据预览：\n{df_clean.head()}")

if __name__ == "__main__":
    main()
