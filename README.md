# 知识图谱工具集

基于本地大模型的知识图谱生成、可视化与查询工具。

## 文件说明

| 文件 | 功能 |
|------|------|
| `文本知识图谱生成器.py` | GUI 工具，从 Markdown/文本文件生成知识图谱 |
| `项目知识图谱生成器.py` | 从项目代码文件（多语言）生成知识图谱 |
| `kg-export-html.py` | 将知识图谱导出为 HTML 可视化文件 |
| `kg-query.py` | 自然语言查询知识图谱 |

## 安装依赖

```bash
pip install openai chromadb rich markdown
```

## 本地模型配置

### LM Studio 配置

1. 安装 [LM Studio](https://lmstudio.ai/)
2. 下载以下模型：
   - **对话模型**: `google/gemma-4-12b-qat`（用于知识图谱提取）
   - **嵌入模型**: `nomic-embed-text-v2-moe`（用于向量化）
3. 启动本地服务器（默认端口 1234）

### 模型参数

| 模型 | 用途 | 参数 |
|------|------|------|
| `google/gemma-4-12b-qat` | 实体关系提取 | temperature=0.1, max_tokens=4096 |
| `nomic-embed-text-v2-moe` | 文本向量化 | 用于 ChromaDB 存储 |

### 服务地址

默认配置：`http://127.0.0.1:1234/v1`

## 使用方法

### 1. 文本知识图谱生成

```bash
python 文本知识图谱生成器.py
```

- 启动 GUI 界面
- 选择 Markdown/文本文件
- 自动提取实体关系并存入 ChromaDB

### 2. 项目代码知识图谱生成

```bash
python 项目知识图谱生成器.py
```

- 弹窗选择项目目录
- 支持增量更新（检测 `.git` 目录）
- 解析多种编程语言（Python, JS, TS, Java, C++, Go, Rust 等）
- 可选 LLM 深度分析

### 3. 知识图谱可视化

```bash
python kg-export-html.py
```

- 导出 `kg_visualization.html`
- 使用 Vis.js 网络图展示
- 支持节点拖拽、缩放、搜索

### 4. 知识图谱查询

```bash
python kg-query.py
```

- 输入自然语言问题
- 向量相似度搜索
- LLM 总结答案

## 数据存储

| 目录 | 内容 |
|------|------|
| `./kg_db/` | ChromaDB 知识图谱数据库 |
| `./kg_data/` | JSON 格式的解析结果 |

## 输出格式

### 三元组结构

```json
{
  "source": "实体A",
  "target": "实体B",
  "relation": "关系描述",
  "desc": "原文摘要"
}
```

### 代码块结构

```json
{
  "type": "function",
  "name": "foo",
  "signature": "(x: int) -> str",
  "docstring": "...",
  "parameters": [{"name": "x", "type": "int"}],
  "return_type": "str",
  "calls": ["bar", "baz"]
}
```

## 支持的编程语言

| 语言 | 解析方式 |
|------|----------|
| Python | AST 模块 |
| JavaScript/TypeScript | 正则匹配 |
| Java/C# | 正则匹配 |
| C/C++ | 正则匹配 |
| Go | 正则匹配 |
| Rust | 正则匹配 |
| Markdown | 标题分块 |

## AI 使用指南

### 快速理解项目流程

当用户请求理解项目时，AI 应按以下步骤操作：

1. **检查知识图谱是否存在**
   - 查看 `./kg_db/` 目录是否存在
   - 查看 `./kg_data/` 目录中的 JSON 文件

2. **读取 JSON 解析结果**
   - 使用 Read 工具读取 `kg_data/kg_output_*.json`
   - 快速获取：文件列表、代码块数量、函数/类定义、调用关系

3. **查询知识图谱（可选）**
   - 运行 `kg-query.py` 进行自然语言查询
   - 或直接读取 ChromaDB 数据

### 推荐查询示例

| 用户问题 | AI 操作 |
|----------|---------|
| "项目有哪些主要模块？" | 读取 JSON，提取 class 类型的代码块 |
| "这个函数做什么？" | 查询函数名，获取 docstring 和调用关系 |
| "谁调用了这个方法？" | 查询 `calls` 字段或三元组中的调用关系 |
| "项目依赖什么库？" | 查询 import 语句或依赖关系三元组 |
| "这个类的继承关系？" | 查询继承/实现类型的三元组 |

### AI Skill 示例代码

```python
# 读取知识图谱 JSON
import json
kg_file = "./kg_data/kg_output_xxx.json"
with open(kg_file, 'r') as f:
    data = json.load(f)

# 统计项目结构
total_files = len(data)
total_chunks = sum(len(d['chunks']) for d in data)
classes = [c for d in data for c in d['chunks'] if c['type'] == 'class']
functions = [c for d in data for c in d['chunks'] if c['type'] == 'function']

# 查找特定函数
func_name = "load_data"
for d in data:
    for c in d['chunks']:
        if c['name'] == func_name:
            print(f"文件: {d['file_path']}")
            print(f"签名: {c['signature']}")
            print(f"调用: {c['calls']}")
```

### 知识图谱查询示例

```python
# 使用 kg-query.py 查询
from openai import OpenAI
import chromadb

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
db = chromadb.PersistentClient(path="./kg_db")
collection = db.get_or_create_collection(name="knowledge_graph")

# 向量搜索
question = "项目的核心功能是什么？"
emb = client.embeddings.create(model="nomic-embed-text-v2-moe", input=[question])
results = collection.query(query_embeddings=[emb.data[0].embedding], n_results=10)

# 返回相关三元组
for meta in results['metadatas'][0]:
    print(f"{meta['source']} → {meta['relation']} → {meta['target']}")
```

### AI 使用场景

| 场景 | 知识图谱优势 |
|------|-------------|
| 代码审查 | 快速定位函数调用链，发现潜在问题 |
| 重构建议 | 分析依赖关系，识别耦合点 |
| 文档生成 | 从 docstring 和签名自动生成文档 |
| Bug 定位 | 查询相关函数，缩小排查范围 |
| 新人引导 | 可视化 HTML 展示项目整体结构 |

### 注意事项

1. **知识图谱可能过期**：项目更新后需重新生成
2. **三元组可能有遗漏**：LLM 提取不保证 100% 准确
3. **结合源码验证**：关键信息应回溯源码确认

## 常见问题

### Q: LM Studio 连接失败？

确保 LM Studio 正在运行，且服务器地址为 `http://127.0.0.1:1234/v1`。

### Q: ChromaDB 报错？

检查 `./kg_db` 目录权限，或删除后重新运行。

### Q: 解析 Python 文件失败？

确保文件语法正确，无编码问题。
