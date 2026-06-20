import os
import json
import hashlib
import subprocess
import time
import ast
import re
from datetime import datetime
from pathlib import Path

try:
    from rich import print
    from rich.panel import Panel
except ImportError:
    # 如果 rich 未安装，使用普通 print
    def print(msg):
        # 简单处理 [color] 标签
        import re as _re
        clean_msg = _re.sub(r'\[/?[a-z_]+\]', '', str(msg))
        __builtins__['print'](clean_msg)
    Panel = lambda text, title: print(f"[{title}] {text}")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    print("警告: tkinter 未安装，将使用命令行输入方式")
    tk = None
    TKINTER_AVAILABLE = False

# ==========================================
# LLM 配置与初始化
# ==========================================
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
LLM_AVAILABLE = True

try:
    from openai import OpenAI
    llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
except ImportError:
    print("[yellow]警告: openai 库未安装，将跳过 LLM 分析功能[/yellow]")
    LLM_AVAILABLE = False

try:
    import chromadb
    db_client = chromadb.PersistentClient(path="./kg_db")
    kg_collection = db_client.get_or_create_collection(name="knowledge_graph")
except ImportError:
    print("[yellow]警告: chromadb 库未安装，将跳过向量化存储功能[/yellow]")
    kg_collection = None

# ==========================================
# LLM 分析辅助函数
# ==========================================
def fix_truncated_json(text):
    """尝试修复被截断的 JSON 数组"""
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

def extract_knowledge_graph(text_chunk):
    """利用 LLM 提取实体和关系"""
    if not LLM_AVAILABLE:
        return []
    
    print(f"[yellow]正在使用 LLM 分析文本块...[/yellow]")
    
    system_prompt = """
    你是一个知识图谱构建专家。请分析用户提供的代码或文本内容，提取其中的实体（节点）和关系（边）。
    请以 JSON 格式输出，不要包含 Markdown 代码块标记。
    JSON 格式要求:
    [
        {"source": "实体A", "target": "实体B", "relation": "关系描述", "desc": "原文摘要"},
        ...
    ]
    注意：
    1. 实体名称要简洁明确
    2. 关系描述要具体准确（如：调用、继承、包含、依赖等）
    3. desc 字段简要说明关系的上下文
    4. 对于代码，关注函数调用、类继承、模块依赖等关系
    """
    
    content = ""
    try:
        response = llm_client.chat.completions.create(
            model="google/gemma-4-12b-qat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分析以下内容并提取图谱数据：\n\n{text_chunk}"}
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
        
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[yellow]标准解析失败，尝试修复截断的 JSON...[/yellow]")
        fixed = fix_truncated_json(content)
        if fixed:
            try:
                return json.loads(fixed)
            except:
                pass
        print(f"[red]解析 JSON 失败: {e}[/red]")
        return []
    except Exception as e:
        print(f"[red]LLM 调用失败: {e}[/red]")
        return []

def generate_triple_id(source, target, relation):
    """为三元组生成一个唯一的 ID"""
    unique_string = f"{source}__{relation}__{target}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def embed_and_store_knowledge_graph(nodes_edges):
    """向量化并存入数据库，自动去重"""
    if not nodes_edges or kg_collection is None:
        return
    
    print(f"[cyan]正在进行向量化存储...[/cyan]")
    
    # 1. 先从数据库中获取所有已存在的 ID，用于去重
    existing_items = kg_collection.get(include=['metadatas'])
    existing_ids = set()
    if existing_items['metadatas']:
        for meta in existing_items['metadatas']:
            doc_id = generate_triple_id(meta['source'], meta['target'], meta['relation'])
            existing_ids.add(doc_id)
    
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    new_items_count = 0
    
    # 2. 用于跟踪当前批次中的 ID，避免批次内重复
    current_batch_ids = set()
    
    for item in nodes_edges:
        # 为当前三元组生成 ID
        doc_id = generate_triple_id(item['source'], item['target'], item['relation'])
        
        # 如果 ID 已存在（数据库中或当前批次中），则跳过
        if doc_id in existing_ids or doc_id in current_batch_ids:
            continue
        
        text_to_embed = f"{item['source']} {item['relation']} {item['target']}. {item['desc']}"
        
        try:
            emb_response = llm_client.embeddings.create(
                model="nomic-embed-text-v2-moe",
                input=[text_to_embed]
            )
            vector = emb_response.data[0].embedding
            
            ids.append(doc_id)
            documents.append(text_to_embed)
            metadatas.append({
                "source": item['source'],
                "target": item['target'],
                "relation": item['relation']
            })
            embeddings.append(vector)
            current_batch_ids.add(doc_id)  # 记录当前批次的 ID
            new_items_count += 1
        except Exception as e:
            print(f"[red]向量化失败: {e}[/red]")
    
    # 3. 批量存入 ChromaDB
    if ids:
        kg_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[green]成功存入 {new_items_count} 条新的知识图谱关系！[/green]")
    else:
        print("[blue]没有新的关系需要存储，所有数据均已存在。[/blue]")

# ==========================================
# 配置常量
# ==========================================
SOURCE_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.h': 'cpp',
    '.cs': 'csharp',
    '.go': 'go',
    '.rs': 'rust',
    '.md': 'markdown',
    '.txt': 'text',
    '.json': 'json',
    '.yml': 'yaml',
    '.yaml': 'yaml',
    '.xml': 'xml',
    '.html': 'html',
    '.css': 'css',
    '.sql': 'sql',
    '.sh': 'shell',
    '.bash': 'shell',
}

IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', '.idea', '.vscode', 'dist', 'build', 'venv', 'env'}
IGNORE_FILES = {'__init__.py', '.gitignore', 'package-lock.json', 'yarn.lock'}


class FileManager:
    """文件管理器：负责扫描项目目录，记录元数据，支持增量更新"""
    
    def __init__(self):
        self.scan_history = {}
        self.git_cache = {}
    
    def select_directory(self):
        """弹窗选择目录"""
        if TKINTER_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            directory = filedialog.askdirectory(title="选择项目根目录")
            root.destroy()
            return directory
        else:
            return input("请输入项目目录路径: ").strip()
    
    def get_language(self, file_path):
        """根据文件扩展名判断语言"""
        ext = Path(file_path).suffix.lower()
        return SOURCE_EXTENSIONS.get(ext, 'unknown')
    
    def get_file_metadata(self, file_path):
        """获取文件元数据"""
        file_stat = os.stat(file_path)
        return {
            'path': file_path,
            'size': file_stat.st_size,
            'language': self.get_language(file_path),
            'last_modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
        }
    
    def is_ignored(self, path):
        """检查是否应该忽略该路径"""
        parts = Path(path).parts
        for part in parts:
            if part in IGNORE_DIRS:
                return True
        return False
    
    def scan_directory(self, root_dir, incremental=False):
        """扫描目录，支持增量更新"""
        files_info = []
        git_diff_files = set()
        
        # 检查是否支持增量更新
        has_git = os.path.isdir(os.path.join(root_dir, '.git'))
        has_pycache = False
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 过滤忽略的目录
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                # 跳过被忽略的文件和路径
                if filename in IGNORE_FILES or self.is_ignored(file_path):
                    continue
                
                # 检查文件扩展名
                ext = Path(filename).suffix.lower()
                if ext not in SOURCE_EXTENSIONS:
                    continue
                
                files_info.append(self.get_file_metadata(file_path))
        
        if incremental and has_git:
            git_diff_files = self.get_git_diff_files(root_dir)
            files_info = [f for f in files_info if f['path'] in git_diff_files]
            print(f"[yellow]增量扫描模式：检测到 {len(files_info)} 个新增/修改的文件[/yellow]")
        else:
            print(f"[cyan]全量扫描模式：共发现 {len(files_info)} 个源代码文件[/cyan]")
        
        return files_info
    
    def get_git_diff_files(self, root_dir):
        """获取 git diff 中新增/修改的文件列表"""
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', '--cached'],
                cwd=root_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            staged_files = set()
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        staged_files.add(os.path.join(root_dir, line.strip()))
            
            # 检查未暂存的修改
            result2 = subprocess.run(
                ['git', 'diff', '--name-only'],
                cwd=root_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result2.returncode == 0:
                for line in result2.stdout.strip().split('\n'):
                    if line:
                        staged_files.add(os.path.join(root_dir, line.strip()))
            
            return staged_files
        except Exception as e:
            print(f"[yellow]获取 git diff 失败，回退到全量扫描: {e}[/yellow]")
            return set()


class CodeParser:
    """代码解析器：针对不同语言提取结构化信息"""
    
    def __init__(self):
        self.parsers = {
            'python': self._parse_python,
            'javascript': self._parse_js_ts,
            'typescript': self._parse_js_ts,
            'java': self._parse_java,
            'cpp': self._parse_cpp,
            'csharp': self._parse_csharp,
            'go': self._parse_go,
            'rust': self._parse_rust,
            'markdown': self._parse_markdown,
            'text': self._parse_text,
            'json': self._parse_json,
        }
    
    def parse_file(self, file_path, language):
        """解析单个文件"""
        parser = self.parsers.get(language, self._parse_default)
        return parser(file_path)
    
    def _parse_python(self, file_path):
        """使用 AST 解析 Python 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    chunk = self._extract_python_chunk(node, lines, 'function')
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    chunk = self._extract_python_chunk(node, lines, 'class')
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.AsyncFunctionDef):
                    chunk = self._extract_python_chunk(node, lines, 'function')
                    if chunk:
                        chunks.append(chunk)
            
            return {'file_path': file_path, 'chunks': chunks}
        except Exception as e:
            print(f"[yellow]解析 Python 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _extract_python_chunk(self, node, lines, node_type):
        """提取 Python AST 节点的信息"""
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        code_snippet = '\n'.join(lines[start_line-1:end_line])
        name = node.name
        
        # 提取参数
        parameters = []
        if hasattr(node, 'args'):
            for arg in node.args.args:
                param_name = arg.arg
                param_type = None
                if arg.annotation:
                    param_type = self._get_annotation_name(arg.annotation)
                parameters.append({'name': param_name, 'type': param_type})
        
        # 提取返回类型
        return_type = None
        if hasattr(node, 'returns') and node.returns:
            return_type = self._get_annotation_name(node.returns)
        
        # 提取 docstring
        docstring = ast.get_docstring(node)
        
        # 提取装饰器
        decorators = []
        if hasattr(node, 'decorator_list'):
            for dec in node.decorator_list:
                decorators.append(self._get_node_name(dec))
        
        # 提取调用的函数
        calls = []
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                calls.append(sub_node.func.id)
        
        # 提取签名
        signature = self._build_signature(name, parameters, return_type)
        
        return {
            'start_line': start_line,
            'end_line': end_line,
            'code_snippet': code_snippet,
            'type': node_type,
            'name': name,
            'signature': signature,
            'docstring': docstring,
            'parameters': parameters,
            'return_type': return_type,
            'decorators': decorators,
            'raises': [],
            'calls': list(set(calls))
        }
    
    def _get_annotation_name(self, node):
        """获取注解的名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            return self._get_annotation_name(node.value)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return None
    
    def _get_node_name(self, node):
        """获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        return str(node)
    
    def _build_signature(self, name, parameters, return_type):
        """构建函数签名"""
        params = []
        for p in parameters:
            if p['type']:
                params.append(f"{p['name']}: {p['type']}")
            else:
                params.append(p['name'])
        params_str = ', '.join(params)
        return_type_str = f" -> {return_type}" if return_type else ""
        return f"{name}({params_str}){return_type_str}"
    
    def _parse_js_ts(self, file_path):
        """解析 JavaScript/TypeScript 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            
            # 使用简单的正则匹配提取函数和类
            
            # 匹配函数声明
            func_pattern = r'(async\s+)?function\s+(\w+)\s*\(([^)]*)\)'
            for match in re.finditer(func_pattern, content):
                async_keyword = match.group(1) or ''
                func_name = match.group(2)
                params_str = match.group(3)
                
                # 查找函数定义的行号
                line_num = content.count('\n', 0, match.start()) + 1
                end_line = self._find_block_end(lines, line_num - 1, '{')
                
                parameters = self._parse_js_parameters(params_str)
                
                signature = f"{async_keyword}function {func_name}({params_str})"
                
                code_snippet = '\n'.join(lines[line_num-1:end_line]) if end_line else '\n'.join(lines[line_num-1:line_num+5])
                
                chunks.append({
                    'start_line': line_num,
                    'end_line': end_line or line_num + 5,
                    'code_snippet': code_snippet,
                    'type': 'function',
                    'name': func_name,
                    'signature': signature,
                    'docstring': None,
                    'parameters': parameters,
                    'return_type': None,
                    'raises': [],
                    'calls': []
                })
            
            # 匹配类声明
            class_pattern = r'class\s+(\w+)\s*(extends\s+\w+)?\s*{'
            for match in re.finditer(class_pattern, content):
                class_name = match.group(1)
                line_num = content.count('\n', 0, match.start()) + 1
                end_line = self._find_block_end(lines, line_num - 1, '{')
                
                code_snippet = '\n'.join(lines[line_num-1:end_line]) if end_line else '\n'.join(lines[line_num-1:line_num+10])
                
                chunks.append({
                    'start_line': line_num,
                    'end_line': end_line or line_num + 10,
                    'code_snippet': code_snippet,
                    'type': 'class',
                    'name': class_name,
                    'signature': match.group(0),
                    'docstring': None,
                    'parameters': [],
                    'return_type': None,
                    'raises': [],
                    'calls': []
                })
            
            return {'file_path': file_path, 'chunks': chunks}
        except Exception as e:
            print(f"[yellow]解析 JS/TS 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _find_block_end(self, lines, start_line, bracket):
        """查找代码块的结束位置"""
        depth = 0
        for i in range(start_line, len(lines)):
            line = lines[i]
            for char in line:
                if char == bracket:
                    depth += 1
                elif char == '}' and bracket == '{':
                    depth -= 1
                    if depth == 0:
                        return i + 1
                elif char == ')' and bracket == '(':
                    depth -= 1
                    if depth == 0:
                        return i + 1
        return None
    
    def _parse_js_parameters(self, params_str):
        """解析 JavaScript 参数"""
        params = []
        if not params_str.strip():
            return params
        
        for param in params_str.split(','):
            param = param.strip()
            if param:
                # 处理带类型注解的参数
                if ':' in param:
                    name, type_name = param.split(':', 1)
                    params.append({'name': name.strip(), 'type': type_name.strip()})
                else:
                    params.append({'name': param, 'type': None})
        return params
    
    def _parse_java(self, file_path):
        """解析 Java 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            
            # 匹配类声明
            class_pattern = r'(public\s+)?class\s+(\w+)\s*(extends\s+\w+)?\s*(implements\s+[\w,]+)?\s*{'
            for match in re.finditer(class_pattern, content):
                class_name = match.group(2)
                line_num = content.count('\n', 0, match.start()) + 1
                end_line = self._find_block_end(lines, line_num - 1, '{')
                
                code_snippet = '\n'.join(lines[line_num-1:end_line]) if end_line else '\n'.join(lines[line_num-1:line_num+10])
                
                chunks.append({
                    'start_line': line_num,
                    'end_line': end_line or line_num + 10,
                    'code_snippet': code_snippet,
                    'type': 'class',
                    'name': class_name,
                    'signature': match.group(0),
                    'docstring': None,
                    'parameters': [],
                    'return_type': None,
                    'raises': [],
                    'calls': []
                })
            
            # 匹配方法声明
            method_pattern = r'(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)'
            for match in re.finditer(method_pattern, content):
                return_type = match.group(3)
                method_name = match.group(4)
                params_str = match.group(5)
                
                line_num = content.count('\n', 0, match.start()) + 1
                end_line = self._find_block_end(lines, line_num - 1, '{')
                
                parameters = []
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param:
                            parts = param.split()
                            if len(parts) >= 2:
                                params.append({'name': parts[-1], 'type': ' '.join(parts[:-1])})
                
                signature = f"{match.group(1) or ''} {match.group(2) or ''} {return_type} {method_name}({params_str})".strip()
                
                code_snippet = '\n'.join(lines[line_num-1:end_line]) if end_line else '\n'.join(lines[line_num-1:line_num+5])
                
                chunks.append({
                    'start_line': line_num,
                    'end_line': end_line or line_num + 5,
                    'code_snippet': code_snippet,
                    'type': 'method',
                    'name': method_name,
                    'signature': signature,
                    'docstring': None,
                    'parameters': parameters,
                    'return_type': return_type,
                    'raises': [],
                    'calls': []
                })
            
            return {'file_path': file_path, 'chunks': chunks}
        except Exception as e:
            print(f"[yellow]解析 Java 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _parse_cpp(self, file_path):
        """解析 C/C++ 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            
            # 匹配函数声明
            func_pattern = r'(\w+\s+)+(\w+)\s*\(([^)]*)\)\s*\{?'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(2)
                params_str = match.group(3)
                
                # 跳过类定义和其他模式
                if 'class' in match.group(1) or 'struct' in match.group(1):
                    continue
                
                line_num = content.count('\n', 0, match.start()) + 1
                end_line = self._find_block_end(lines, line_num - 1, '{')
                
                parameters = []
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and '=' not in param:
                            parts = param.split()
                            if len(parts) >= 2:
                                parameters.append({'name': parts[-1], 'type': ' '.join(parts[:-1])})
                
                signature = f"{match.group(1)}{func_name}({params_str})".strip()
                
                code_snippet = '\n'.join(lines[line_num-1:end_line]) if end_line else '\n'.join(lines[line_num-1:line_num+5])
                
                chunks.append({
                    'start_line': line_num,
                    'end_line': end_line or line_num + 5,
                    'code_snippet': code_snippet,
                    'type': 'function',
                    'name': func_name,
                    'signature': signature,
                    'docstring': None,
                    'parameters': parameters,
                    'return_type': None,
                    'raises': [],
                    'calls': []
                })
            
            return {'file_path': file_path, 'chunks': chunks}
        except Exception as e:
            print(f"[yellow]解析 C/C++ 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _parse_csharp(self, file_path):
        """解析 C# 文件"""
        return self._parse_java(file_path)  # C# 语法与 Java 类似
    
    def _parse_go(self, file_path):
        """解析 Go 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            
            # 匹配函数声明
            func_pattern = r'func\s+(\w+)\s*\(([^)]*)\)\s*(\w+)?\s*{'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                params_str = match.group(2)
                return_type = match.group(3)
                
                line_num = content.count('\n', 0, match.start()) + 1
                end_line = self._find_block_end(lines, line_num - 1, '{')
                
                parameters = []
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param:
                            parts = param.split()
                            if len(parts) >= 2:
                                parameters.append({'name': parts[0], 'type': parts[1]})
                
                signature = f"func {func_name}({params_str})"
                if return_type:
                    signature += f" {return_type}"
                
                code_snippet = '\n'.join(lines[line_num-1:end_line]) if end_line else '\n'.join(lines[line_num-1:line_num+5])
                
                chunks.append({
                    'start_line': line_num,
                    'end_line': end_line or line_num + 5,
                    'code_snippet': code_snippet,
                    'type': 'function',
                    'name': func_name,
                    'signature': signature,
                    'docstring': None,
                    'parameters': parameters,
                    'return_type': return_type,
                    'raises': [],
                    'calls': []
                })
            
            return {'file_path': file_path, 'chunks': chunks}
        except Exception as e:
            print(f"[yellow]解析 Go 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _parse_rust(self, file_path):
        """解析 Rust 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            
            # 匹配函数声明
            func_pattern = r'fn\s+(\w+)\s*\(([^)]*)\)\s*(->\s*\w+)?\s*{'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                params_str = match.group(2)
                return_type = match.group(3)
                
                line_num = content.count('\n', 0, match.start()) + 1
                end_line = self._find_block_end(lines, line_num - 1, '{')
                
                parameters = []
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param:
                            parts = param.split(':')
                            if len(parts) >= 2:
                                parameters.append({'name': parts[0].strip(), 'type': parts[1].strip()})
                
                signature = f"fn {func_name}({params_str})"
                if return_type:
                    signature += return_type
                
                code_snippet = '\n'.join(lines[line_num-1:end_line]) if end_line else '\n'.join(lines[line_num-1:line_num+5])
                
                chunks.append({
                    'start_line': line_num,
                    'end_line': end_line or line_num + 5,
                    'code_snippet': code_snippet,
                    'type': 'function',
                    'name': func_name,
                    'signature': signature,
                    'docstring': None,
                    'parameters': parameters,
                    'return_type': return_type.strip('-> ').strip() if return_type else None,
                    'raises': [],
                    'calls': []
                })
            
            return {'file_path': file_path, 'chunks': chunks}
        except Exception as e:
            print(f"[yellow]解析 Rust 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _parse_markdown(self, file_path):
        """解析 Markdown 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            
            # 按标题分块
            current_chunk = []
            current_title = None
            start_line = 1
            
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    if current_chunk:
                        chunks.append({
                            'start_line': start_line,
                            'end_line': i,
                            'code_snippet': '\n'.join(current_chunk),
                            'type': 'section',
                            'name': current_title or 'Untitled',
                            'signature': None,
                            'docstring': '\n'.join(current_chunk),
                            'parameters': [],
                            'return_type': None,
                            'raises': [],
                            'calls': []
                        })
                    current_title = line.strip('#').strip()
                    current_chunk = []
                    start_line = i + 1
                else:
                    current_chunk.append(line)
            
            if current_chunk:
                chunks.append({
                    'start_line': start_line,
                    'end_line': len(lines),
                    'code_snippet': '\n'.join(current_chunk),
                    'type': 'section',
                    'name': current_title or 'Untitled',
                    'signature': None,
                    'docstring': '\n'.join(current_chunk),
                    'parameters': [],
                    'return_type': None,
                    'raises': [],
                    'calls': []
                })
            
            return {'file_path': file_path, 'chunks': chunks}
        except Exception as e:
            print(f"[yellow]解析 Markdown 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _parse_text(self, file_path):
        """解析纯文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            return {
                'file_path': file_path,
                'chunks': [{
                    'start_line': 1,
                    'end_line': len(lines),
                    'code_snippet': content,
                    'type': 'text',
                    'name': Path(file_path).stem,
                    'signature': None,
                    'docstring': content,
                    'parameters': [],
                    'return_type': None,
                    'raises': [],
                    'calls': []
                }]
            }
        except Exception as e:
            print(f"[yellow]解析文本文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _parse_json(self, file_path):
        """解析 JSON 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = json.loads(content)
            
            return {
                'file_path': file_path,
                'chunks': [{
                    'start_line': 1,
                    'end_line': len(content.split('\n')),
                    'code_snippet': content,
                    'type': 'data',
                    'name': Path(file_path).stem,
                    'signature': None,
                    'docstring': json.dumps(data, indent=2, ensure_ascii=False),
                    'parameters': [],
                    'return_type': None,
                    'raises': [],
                    'calls': []
                }]
            }
        except Exception as e:
            print(f"[yellow]解析 JSON 文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}
    
    def _parse_default(self, file_path):
        """默认解析器：返回文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            return {
                'file_path': file_path,
                'chunks': [{
                    'start_line': 1,
                    'end_line': len(lines),
                    'code_snippet': content,
                    'type': 'unknown',
                    'name': Path(file_path).stem,
                    'signature': None,
                    'docstring': None,
                    'parameters': [],
                    'return_type': None,
                    'raises': [],
                    'calls': []
                }]
            }
        except Exception as e:
            print(f"[yellow]解析文件失败 {file_path}: {e}[/yellow]")
            return {'file_path': file_path, 'chunks': []}


class KnowledgeGraphGenerator:
    """项目知识图谱生成器"""
    
    def __init__(self):
        self.file_manager = FileManager()
        self.code_parser = CodeParser()
        self.output_dir = './kg_data'
    
    def run(self):
        """主入口函数"""
        print("=" * 60)
        print("        项目知识图谱生成器")
        print("=" * 60)
        
        # 1. 选择目录
        project_dir = self.file_manager.select_directory()
        if not project_dir:
            print("[red]错误：未选择目录[/red]")
            return
        
        if not os.path.isdir(project_dir):
            print(f"[red]错误：目录不存在 {project_dir}[/red]")
            return
        
        print(f"[green]已选择项目目录: {project_dir}[/green]")
        
        # 2. 询问是否增量更新
        use_incremental = False
        has_git = os.path.isdir(os.path.join(project_dir, '.git'))
        if has_git:
            if TKINTER_AVAILABLE:
                root = tk.Tk()
                root.withdraw()
                result = messagebox.askyesno("增量更新", "检测到 .git 目录，是否启用增量更新模式？")
                root.destroy()
                use_incremental = result
            else:
                choice = input("检测到 .git 目录，是否启用增量更新模式？(y/n): ").strip().lower()
                use_incremental = choice == 'y'
        
        # 3. 扫描文件
        print("\n[blue]开始扫描项目文件...[/blue]")
        files_info = self.file_manager.scan_directory(project_dir, incremental=use_incremental)
        
        if not files_info:
            print("[yellow]未发现需要处理的文件[/yellow]")
            return
        
        # 4. 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 5. 解析文件并生成图谱数据
        all_results = []
        for i, file_info in enumerate(files_info):
            file_path = file_info['path']
            language = file_info['language']
            
            print(f"\n[cyan]处理文件 {i+1}/{len(files_info)}: {file_path}[/cyan]")
            
            # 解析文件
            result = self.code_parser.parse_file(file_path, language)
            
            # 添加元数据
            result['metadata'] = file_info
            all_results.append(result)
            
            # 显示解析结果摘要
            num_chunks = len(result['chunks'])
            if num_chunks > 0:
                print(f"  [green]✓ 提取到 {num_chunks} 个代码块[/green]")
                for chunk in result['chunks'][:3]:
                    print(f"    - {chunk['type']}: {chunk['name']} (行 {chunk['start_line']}-{chunk['end_line']})")
        
        # 6. 保存结果
        output_file = os.path.join(self.output_dir, f"kg_output_{int(time.time())}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print(f"[bold blue]✅ 代码解析完成！[/bold blue]")
        print(f"[green]输出文件: {output_file}[/green]")
        print(f"[green]处理文件数: {len(files_info)}[/green]")
        total_chunks = sum(len(r['chunks']) for r in all_results)
        print(f"[green]提取代码块数: {total_chunks}[/green]")
        print("=" * 60)
        
        # 7. 询问是否使用 LLM 进行深度知识图谱分析
        use_llm = False
        if LLM_AVAILABLE:
            if TKINTER_AVAILABLE:
                root = tk.Tk()
                root.withdraw()
                result = messagebox.askyesno("LLM 深度分析", "是否使用本地大模型进行深度知识图谱分析？\n（此功能需要 LM Studio 运行在 http://127.0.0.1:1234）")
                root.destroy()
                use_llm = result
            else:
                choice = input("是否使用本地大模型进行深度知识图谱分析？(y/n): ").strip().lower()
                use_llm = choice == 'y'
        
        if use_llm:
            print("\n" + "=" * 60)
            print("        开始 LLM 深度分析")
            print("=" * 60)
            
            # 7.1 将解析结果转换为文本，用于 LLM 分析
            all_graph_data = []
            chunk_size = 3000
            current_text = ""
            
            for result in all_results:
                file_path = result['file_path']
                for chunk in result['chunks']:
                    chunk_text = f"文件: {file_path}\n类型: {chunk['type']}\n名称: {chunk['name']}\n签名: {chunk['signature']}\n代码:\n{chunk['code_snippet']}\n\n"
                    
                    if len(current_text) + len(chunk_text) > chunk_size and current_text:
                        # 处理当前文本块
                        graph_data = extract_knowledge_graph(current_text)
                        all_graph_data.extend(graph_data)
                        
                        # 显示部分结果
                        for rel in graph_data[:2]:
                            print(f"🔗 {rel['source']} --({rel['relation']})--> {rel['target']}")
                        
                        current_text = chunk_text
                    else:
                        current_text += chunk_text
            
            # 处理最后一个文本块
            if current_text:
                graph_data = extract_knowledge_graph(current_text)
                all_graph_data.extend(graph_data)
                for rel in graph_data[:2]:
                    print(f"🔗 {rel['source']} --({rel['relation']})--> {rel['target']}")
            
            # 7.2 向量化并存储
            embed_and_store_knowledge_graph(all_graph_data)
            
            print("\n" + "=" * 60)
            print(f"[bold blue]✅ LLM 深度分析完成！[/bold blue]")
            print(f"[green]提取关系数: {len(all_graph_data)}[/green]")
            print(f"[green]数据已保存在 ./kg_db 目录中[/green]")
            print("=" * 60)


if __name__ == "__main__":
    generator = KnowledgeGraphGenerator()
    generator.run()
