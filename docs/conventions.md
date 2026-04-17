AI 文本检测项目——代码与团队协作规范

本规范旨在确保团队（AI 生成文本检测项目组）在 Python 开发与 GitHub 协作时保持高效、统一、可维护。请全员开发前阅读并遵守。

一、Python 代码变量命名规则

本项目遵循精简版 PEP 8 规范，重点保证数据清洗、特征提取（TF-IDF）、模型训练等环节代码可读性一致。

1. 蛇形命名法 (snake_case)

- 适用范围：变量名、函数名、方法名、模块/文件名

- 规则：全部小写，单词之间用下划线 `_` 连接（示例：`cleaned_text`、`extract_tfidf_features()`、`data_preprocessing.py`）

- 数据处理约定：      

  - 原始数据：`df_raw`

  - 清洗后数据：`df_clean`

  - 训练/测试集：`X_train`、`X_test`、`y_train`、`y_test`

- 禁止：拼音、无意义缩写（如 a、tmp、data1）、中文命名

2. 大驼峰命名法 (PascalCase)

- 适用范围：类名 (Class)

- 规则：每个单词首字母大写，不使用下划线（示例：`LogisticRegressionModel`、`TextEvaluator`、`AIDetector`）

3. 全大写命名法 (UPPER_CASE_SNAKE_CASE)

- 适用范围：全局常量、配置项（示例：`MAX_TEXT_LENGTH = 500`、`RANDOM_SEED = 42`、`TFIDF_MAX_FEATURES = 3000`）

4. 私有成员约定

- 类内部私有属性/方法以单下划线开头：`_preprocess_text()`


---
二、GitHub 分支命名规则

严禁直接 push 代码到 `main` 分支，所有开发必须在个人特性分支完成。

- 统一格式：`姓名拼音/类型-简短描述`（全部小写）

- （类型缩写说明：`feat`：新功能；`fix`：修复问题；`docs`：文档；`refactor`：重构；`test`：测试）

- 示例：

  - `xingming/feat-data-cleaning`

  - `xingming/docs-update-spec`

  - `xingming/fix-csv-encoding`


---
三、Commit 提交信息规范

每次提交必须清晰、规范，统一使用 `类型: 描述内容` 格式（冒号后必须加一个空格）。

- feat: 新增功能或模块
例：`feat: 实现 TF-IDF 特征提取函数`

- fix: 修复 Bug、逻辑错误、报错 
例：`fix: 解决 CSV 读取时中文字符编码报错`

- docs: 文档修改、注释更新、规范编写
例：`docs: 完善代码与团队协作规范`

- chore: 构建、依赖、配置、工具调整
例：`chore: 更新 requirements.txt 依赖`

- refactor: 代码重构（不改变功能，优化结构/可读性）
例：`refactor: 拆分数据清洗函数，提高复用性`

- test: 新增或修改测试代码      
例：`test: 添加文本清洗函数的测试用例`

禁止：无意义提交如 `update`、`fix`、`test`、`111`。


---
四、PR (Pull Request) 协作流程

1. 从最新 `main` 或 `dev` 分支切出个人功能分支

2. 本地开发 → 规范 commit → push 到远程

3. 在 GitHub 发起 PR 到主开发分支

4. PR 标题遵循 commit 规范，正文写明：

  - 完成内容

  - 改动点

  - 测试情况

5. 组长审核并 Approve 后方可合并

6. 合并完成后及时删除功能分支，保持仓库整洁


---
五、文件与大文件管理规范

- GitHub 只允许存放：`.py`、`.ipynb`、`README.md`、配置文件、规范文档等轻量文件

- 严禁上传：原始数据集、大 CSV、训练好的模型文件（`.pkl`/`.h5`）、压缩包、日志文件

- 大型文件统一存放：项目 OneDrive 共享文件夹

- 补充说明：如必须使用数据文件，可在代码中使用相对路径读取 OneDrive 同步目录，避免上传到 Git


---
六、统一开发约定

1. 代码缩进统一使用 4 个空格

2. 函数必须添加简要注释，说明功能、输入、输出

3. 同一模块避免重复代码，尽量抽为公共函数

4. 每次合并前确保代码可运行，不把报错代码合入主分支


