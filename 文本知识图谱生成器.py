#!/usr/bin/env python3
"""
Knowledge Graph Generator - GUI
A professional tool to generate knowledge graphs from text files.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
import sys
import json
import markdown
import hashlib
from openai import OpenAI
import chromadb

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LM Studio configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")

# ChromaDB
db_client = chromadb.PersistentClient(path="./kg_db")
collection = db_client.get_or_create_collection(name="knowledge_graph")

# ====================
# 知识图谱配置
# ====================

# 标准化关系类型体系
STANDARD_RELATIONS = [
    "属于", "包含", "组成", "生成", "输入", "输出",
    "执行", "调用", "使用", "依赖", "继承", "实现",
    "功能描述", "作用于", "关联", "位于", "创建", "拥有"
]

# 实体类型分类
ENTITY_TYPES = [
    "概念", "实体", "功能模块", "指令", "程序", "设备",
    "属性", "方法", "类", "接口", "系统", "流程"
]

def read_and_clean_md(file_path):
    """Read MD file and convert to plain text"""
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    text_content = markdown.markdown(md_content, extensions=['extra'])
    # Remove HTML tags
    import re
    clean_text = re.sub(r'<[^<]+?>', '', text_content)
    return clean_text

def fix_truncated_json(text):
    """Try to fix truncated JSON array"""
    text = text.strip()
    if text.startswith('['):
        open_brackets = 0
        in_string = False
        escape_next = False
        last_complete_pos = 0

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    open_brackets += 1
                elif char == '}':
                    open_brackets -= 1
                    if open_brackets == 0:
                        last_complete_pos = i + 1

        if last_complete_pos > 0 and text[last_complete_pos-1] == '}':
            return text[:last_complete_pos] + ']'
    return None

def generate_triple_id(source, target, relation):
    """Generate a unique ID for a triple (source, relation, target)"""
    unique_string = f"{source}__{relation}__{target}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def extract_knowledge_graph(text_chunk, log_callback=None):
    """Extract entities and relations using LLM with professional KG structure"""
    def log(msg):
        if log_callback:
            log_callback(msg)

    system_prompt = """
    你是一位专业的知识图谱构建专家。请分析提供的文本，提取其中的实体（节点）和关系（边）。
    
    输出要求：
    1. 必须是纯JSON数组格式，不要包含任何Markdown代码块标记
    2. 每个条目必须包含以下字段：
       - source: 源实体名称（简洁明确）
       - source_type: 源实体类型（如：概念、实体、功能模块、指令、程序、设备等）
       - target: 目标实体名称（简洁明确）
       - target_type: 目标实体类型（如：概念、实体、功能模块、指令、程序、设备等）
       - relation: 关系描述（请使用标准化类型：属于、包含、组成、生成、输入、输出、执行、调用、使用、依赖、继承、实现、功能描述、作用于、关联、位于、创建、拥有）
       - confidence: 置信度（0.0-1.0之间，表示该关系的可信程度）
       - desc: 原文摘要（简要说明该关系的来源依据）
    
    示例输出格式：
    [
        {"source": "PLC", "source_type": "设备", "target": "控制系统", "target_type": "系统", "relation": "属于", "confidence": 0.95, "desc": "PLC是可编程逻辑控制器，属于工业控制系统的核心组件"},
        {"source": "WDR", "source_type": "指令", "target": "看门狗复位", "target_type": "功能描述", "relation": "功能描述", "confidence": 0.98, "desc": "WDR指令用于执行看门狗复位操作"}
    ]
    
    注意事项：
    - 实体名称要简洁准确，避免冗长
    - 关系描述要使用标准化类型，确保一致性
    - 置信度根据文本证据强度合理分配
    - 只提取明确的、有文本支持的关系
    """

    content = ""
    try:
        response = client.chat.completions.create(
            model="google/gemma-4-12b-qat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分析以下内容并提取知识图谱数据：\n\n{text_chunk}"}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        content = response.choices[0].message.content

        content = content.strip()
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)
        
        # 验证并标准化结果
        validated_result = []
        for item in result:
            # 确保所有必需字段存在
            if all(key in item for key in ['source', 'target', 'relation', 'desc']):
                validated_item = {
                    'source': item['source'],
                    'source_type': item.get('source_type', '实体'),
                    'target': item['target'],
                    'target_type': item.get('target_type', '实体'),
                    'relation': item['relation'],
                    'confidence': item.get('confidence', 0.8),
                    'desc': item['desc']
                }
                validated_result.append(validated_item)
        
        log(f"成功提取 {len(validated_result)} 条有效关系")
        return validated_result
        
    except json.JSONDecodeError:
        log("⚠️ 标准解析失败，尝试修复截断的JSON...")
        fixed = fix_truncated_json(content)
        if fixed:
            try:
                result = json.loads(fixed)
                validated_result = []
                for item in result:
                    if all(key in item for key in ['source', 'target', 'relation', 'desc']):
                        validated_item = {
                            'source': item['source'],
                            'source_type': item.get('source_type', '实体'),
                            'target': item['target'],
                            'target_type': item.get('target_type', '实体'),
                            'relation': item['relation'],
                            'confidence': item.get('confidence', 0.8),
                            'desc': item['desc']
                        }
                        validated_result.append(validated_item)
                log(f"修复成功，提取 {len(validated_result)} 条关系")
                return validated_result
            except Exception as e:
                log(f"❌ 修复失败: {str(e)}")
        log(f"❌ JSON解析失败，原始输出预览: {content[:300]}...")
        return []
    except Exception as e:
        log(f"❌ 提取过程发生错误: {str(e)}")
        return []

def embed_and_store(nodes_edges, log_callback=None):
    """Generate embeddings and store in database with deduplication"""
    def log(msg):
        if log_callback:
            log_callback(msg)

    if not nodes_edges:
        log("⚠️ 没有可存储的数据")
        return

    log("🔄 正在进行向量化存储...")

    # 获取已存在的ID用于去重
    existing_items = collection.get(include=['metadatas'])
    existing_ids = set()
    if existing_items['metadatas']:
        for meta in existing_items['metadatas']:
            if 'source' in meta and 'target' in meta and 'relation' in meta:
                doc_id = generate_triple_id(meta['source'], meta['target'], meta['relation'])
                existing_ids.add(doc_id)
    
    log(f"📊 数据库中已存在 {len(existing_ids)} 条关系")

    ids = []
    documents = []
    metadatas = []
    embeddings = []
    new_count = 0
    duplicate_count = 0

    for item in nodes_edges:
        # 生成唯一ID
        doc_id = generate_triple_id(item['source'], item['target'], item['relation'])
        
        # 去重检查
        if doc_id in existing_ids:
            duplicate_count += 1
            continue

        # 构建向量化文本
        text_to_embed = f"{item['source']}({item['source_type']}) {item['relation']} {item['target']}({item['target_type']}). {item['desc']}"

        try:
            emb_response = client.embeddings.create(
                model="nomic-embed-text-v2-moe",
                input=[text_to_embed]
            )
            vector = emb_response.data[0].embedding

            ids.append(doc_id)
            documents.append(text_to_embed)
            metadatas.append({
                "source": item['source'],
                "source_type": item['source_type'],
                "target": item['target'],
                "target_type": item['target_type'],
                "relation": item['relation'],
                "confidence": item['confidence'],
                "desc": item['desc']
            })
            embeddings.append(vector)
            new_count += 1
        except Exception as e:
            log(f"❌ 向量化失败: {str(e)}")

    # 批量存入数据库
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        log(f"✅ 成功存入 {new_count} 条新知识图谱关系")
        if duplicate_count > 0:
            log(f"ℹ️ 跳过 {duplicate_count} 条重复关系")
    else:
        log("ℹ️ 没有新的关系需要存储，所有数据均已存在")

class KGGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("知识图谱生成器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # 设置窗口图标和样式
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        self.input_file = None
        self.chunk_size = 3000

        self._create_widgets()

    def _create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", pady=15)
        title_frame.pack(fill=tk.X)
        
        title = tk.Label(
            title_frame,
            text="知识图谱生成器",
            font=("Microsoft YaHei", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title.pack()

        subtitle = tk.Label(
            title_frame,
            text="Knowledge Graph Generator",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#bdc3c7"
        )
        subtitle.pack()

        # File selection frame
        file_frame = tk.Frame(self.root, padx=20, pady=15, bg="#ecf0f1")
        file_frame.pack(fill=tk.X)

        self.file_label = tk.Label(
            file_frame, 
            text="未选择文件", 
            fg="#7f8c8d",
            font=("Microsoft YaHei", 11),
            bg="#ecf0f1"
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = tk.Button(
            file_frame,
            text="浏览文件...",
            command=self.browse_file,
            width=12,
            bg="#3498db",
            fg="white",
            font=("Microsoft YaHei", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        browse_btn.bind('<Enter>', lambda e: browse_btn.config(bg="#2980b9"))
        browse_btn.bind('<Leave>', lambda e: browse_btn.config(bg="#3498db"))

        # Options frame
        options_frame = tk.Frame(self.root, padx=20, pady=10, bg="white")
        options_frame.pack(fill=tk.X)

        # Chunk size option
        chunk_frame = tk.Frame(options_frame, bg="white")
        chunk_frame.pack(side=tk.LEFT)
        
        tk.Label(
            chunk_frame, 
            text="分块大小:", 
            font=("Microsoft YaHei", 10),
            bg="white"
        ).pack(side=tk.LEFT)

        self.chunk_size_var = tk.StringVar(value="3000")
        chunk_entry = tk.Entry(
            chunk_frame,
            textvariable=self.chunk_size_var,
            width=10,
            font=("Arial", 10),
            justify=tk.CENTER,
            relief=tk.SUNKEN,
            borderwidth=1
        )
        chunk_entry.pack(side=tk.LEFT, padx=(8, 5))

        tk.Label(
            chunk_frame, 
            text="(字符/批次)", 
            font=("Microsoft YaHei", 9),
            fg="#7f8c8d",
            bg="white"
        ).pack(side=tk.LEFT)

        # Stats button
        stats_btn = tk.Button(
            options_frame,
            text="📊 查看统计",
            command=self.show_stats,
            width=12,
            bg="#9b59b6",
            fg="white",
            font=("Microsoft YaHei", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        stats_btn.pack(side=tk.RIGHT)
        stats_btn.bind('<Enter>', lambda e: stats_btn.config(bg="#8e44ad"))
        stats_btn.bind('<Leave>', lambda e: stats_btn.config(bg="#9b59b6"))

        # Progress area
        progress_frame = tk.Frame(self.root, padx=20, pady=10, bg="white")
        progress_frame.pack(fill=tk.BOTH, expand=True)

        progress_label = tk.Label(
            progress_frame,
            text="处理日志",
            font=("Microsoft YaHei", 11, "bold"),
            bg="white",
            fg="#2c3e50"
        )
        progress_label.pack(anchor=tk.W)

        self.progress_text = scrolledtext.ScrolledText(
            progress_frame,
            height=20,
            font=("Consolas", 10),
            state=tk.DISABLED,
            wrap=tk.WORD,
            relief=tk.SUNKEN,
            borderwidth=1,
            bg="#f8f9fa"
        )
        self.progress_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Control buttons
        btn_frame = tk.Frame(self.root, pady=15, bg="#ecf0f1")
        btn_frame.pack(fill=tk.X)

        self.start_btn = tk.Button(
            btn_frame,
            text="开始生成",
            command=self.start_generation,
            width=15,
            bg="#27ae60",
            fg="white",
            font=("Microsoft YaHei", 12, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        self.start_btn.pack(side=tk.LEFT, padx=(20, 10))
        self.start_btn.bind('<Enter>', lambda e: self.start_btn.config(bg="#2ecc71"))
        self.start_btn.bind('<Leave>', lambda e: self.start_btn.config(bg="#27ae60"))

        clear_btn = tk.Button(
            btn_frame,
            text="清空日志",
            command=self.clear_log,
            width=10,
            bg="#95a5a6",
            fg="white",
            font=("Microsoft YaHei", 10),
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        clear_btn.bind('<Enter>', lambda e: clear_btn.config(bg="#aeb6bf"))
        clear_btn.bind('<Leave>', lambda e: clear_btn.config(bg="#95a5a6"))

        export_btn = tk.Button(
            btn_frame,
            text="导出数据",
            command=self.export_data,
            width=10,
            bg="#f39c12",
            fg="white",
            font=("Microsoft YaHei", 10),
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        export_btn.pack(side=tk.LEFT, padx=5)
        export_btn.bind('<Enter>', lambda e: export_btn.config(bg="#f1c40f"))
        export_btn.bind('<Leave>', lambda e: export_btn.config(bg="#f39c12"))

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="选择文本文件",
            filetypes=[
                ("Markdown文件", "*.md"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.input_file = file_path
            self.file_label.config(text=os.path.basename(file_path), fg="#2c3e50")

    def log(self, message):
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.progress_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def clear_log(self):
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state=tk.DISABLED)

    def show_stats(self):
        """显示数据库统计信息"""
        try:
            collection_stats = collection.count()
            existing_items = collection.get(include=['metadatas'])
            
            # 统计关系类型分布
            relation_counts = {}
            if existing_items['metadatas']:
                for meta in existing_items['metadatas']:
                    relation = meta.get('relation', '未知')
                    relation_counts[relation] = relation_counts.get(relation, 0) + 1
            
            stats_msg = f"📊 数据库统计信息\n"
            stats_msg += "=" * 40 + "\n"
            stats_msg += f"总关系数量: {collection_stats}\n"
            stats_msg += "\n关系类型分布:\n"
            for relation, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
                stats_msg += f"  - {relation}: {count} 条\n"
            
            messagebox.showinfo("统计信息", stats_msg)
        except Exception as e:
            messagebox.showerror("错误", f"获取统计信息失败: {str(e)}")

    def export_data(self):
        """导出知识图谱数据"""
        try:
            existing_items = collection.get(include=['metadatas', 'documents'])
            
            if not existing_items['metadatas']:
                messagebox.showwarning("提示", "数据库中没有数据")
                return
            
            export_data = []
            for meta, doc in zip(existing_items['metadatas'], existing_items['documents']):
                export_item = {
                    'source': meta.get('source', ''),
                    'source_type': meta.get('source_type', ''),
                    'target': meta.get('target', ''),
                    'target_type': meta.get('target_type', ''),
                    'relation': meta.get('relation', ''),
                    'confidence': meta.get('confidence', 0.0),
                    'desc': meta.get('desc', ''),
                    'document': doc
                }
                export_data.append(export_item)
            
            export_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json")],
                title="导出知识图谱数据"
            )
            
            if export_path:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", f"数据已导出到:\n{export_path}")
        
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def start_generation(self):
        if not self.input_file:
            messagebox.showwarning("警告", "请先选择文件！")
            return

        if not os.path.exists(self.input_file):
            messagebox.showerror("错误", "文件不存在！")
            return

        try:
            self.chunk_size = int(self.chunk_size_var.get())
            if self.chunk_size < 500 or self.chunk_size > 10000:
                messagebox.showerror("错误", "分块大小应在 500-10000 之间！")
                return
        except ValueError:
            messagebox.showerror("错误", "无效的分块大小！")
            return

        # Disable button
        self.start_btn.config(state=tk.DISABLED, text="处理中...", bg="#95a5a6")

        # Run in thread
        thread = threading.Thread(target=self._process_file)
        thread.daemon = True
        thread.start()

    def _process_file(self):
        try:
            self.log("=" * 60)
            self.log(f"📁 正在读取文件: {os.path.basename(self.input_file)}")
            raw_text = read_and_clean_md(self.input_file)
            self.log(f"📄 文件加载完成，总字符数: {len(raw_text):,}")

            # Split into chunks
            chunks = [raw_text[i:i+self.chunk_size] for i in range(0, len(raw_text), self.chunk_size)]
            self.log(f"🔪 已分割为 {len(chunks)} 个文本块")
            self.log("=" * 60)

            all_graph_data = []
            total_extracted = 0

            for i, chunk in enumerate(chunks):
                self.log(f"\n📊 处理文本块 {i+1}/{len(chunks)}")
                self.log("正在分析并提取知识图谱...")
                
                graph_data = extract_knowledge_graph(chunk, log_callback=self.log)
                all_graph_data.extend(graph_data)
                total_extracted += len(graph_data)
                
                # 显示部分提取结果
                if graph_data:
                    self.log("\n提取的关系示例:")
                    for rel in graph_data[:3]:
                        self.log(f"  🔗 {rel['source']}({rel['source_type']}) --({rel['relation']})--> {rel['target']}({rel['target_type']}) [置信度: {rel['confidence']}]")

            self.log("\n" + "=" * 60)
            self.log(f"📈 提取完成！共提取 {total_extracted} 条关系")
            self.log("💾 正在存储到数据库...")

            embed_and_store(all_graph_data, log_callback=self.log)

            self.log("\n" + "=" * 60)
            self.log("🎉 知识图谱构建完成！")
            self.log(f"📁 数据已保存到 ./kg_db 目录")
            self.log("=" * 60)
            self.log("\n💡 提示：您可以点击「查看统计」查看数据库状态")
            self.log("💡 提示：您可以点击「导出数据」导出JSON格式的图谱数据")

        except Exception as e:
            self.log(f"\n❌ 错误: {str(e)}")
            messagebox.showerror("错误", str(e))

        finally:
            # Re-enable button
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL, text="开始生成", bg="#27ae60"))

def main():
    root = tk.Tk()
    app = KGGeneratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()