import json
from openai import OpenAI
import chromadb
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ==========================================
# 配置与初始化
# ==========================================
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")

# 连接 ChromaDB
db_client = chromadb.PersistentClient(path="./kg_db")
collection = db_client.get_or_create_collection(name="knowledge_graph")

console = Console()

def get_embedding(text):
    """获取文本嵌入向量"""
    response = client.embeddings.create(
        model="nomic-embed-text-v2-moe",
        input=[text]
    )
    return response.data[0].embedding

def query_knowledge_graph(question, top_k=10):
    """
    自然语言查询知识图谱
    """
    console.print(f"[yellow]正在分析问题...[/yellow]")

    # 1. 将问题转换为嵌入向量
    question_embedding = get_embedding(question)

    # 2. 在 ChromaDB 中搜索最相关的关系
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    if not results or not results['documents']:
        console.print("[red]未找到相关结果[/red]")
        return []

    # 3. 整理结果
    relations = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        relations.append({
            "source": metadata['source'],
            "relation": metadata['relation'],
            "target": metadata['target'],
            "desc": doc,
            "distance": results['distances'][0][i]
        })

    return relations

def display_results(relations, question):
    """以表格形式展示查询结果"""
    console.print(f"\n[bold cyan]问题:[/bold cyan] {question}\n")

    if not relations:
        console.print("[red]没有找到相关知识图谱关系[/red]")
        return

    # 按相关性排序（距离越小越相关）
    relations.sort(key=lambda x: x['distance'])

    table = Table(title=f"找到 {len(relations)} 条相关关系", show_header=True)
    table.add_column("来源实体", style="cyan", no_wrap=False)
    table.add_column("关系", style="yellow")
    table.add_column("目标实体", style="green", no_wrap=False)
    table.add_column("描述", style="dim")

    for rel in relations:
        # 简化描述，只显示核心内容
        desc = rel['desc']
        # 从描述中提取有用部分
        table.add_row(rel['source'], rel['relation'], rel['target'], desc[:50] + "..." if len(desc) > 50 else desc)

    console.print(table)

def ask_about_entity(entity_name, top_k=20):
    """查询与特定实体相关的所有关系"""
    console.print(f"\n[bold cyan]查询实体:[/bold cyan] {entity_name}\n")

    question_embedding = get_embedding(entity_name)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    if not results or not results['documents']:
        console.print("[red]未找到相关结果[/red]")
        return []

    relations = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        relations.append({
            "source": metadata['source'],
            "relation": metadata['relation'],
            "target": metadata['target'],
            "desc": doc,
            "distance": results['distances'][0][i]
        })

    return relations

def interactive_mode():
    """交互式查询模式"""
    console.print(Panel.fit(
        "[bold]知识图谱查询系统[/bold]\n"
        "输入问题进行查询，或输入实体名称（以 @ 开头）查看相关关系\n"
        "输入 [red]quit[/red] 退出",
        title="帮助"
    ))

    while True:
        try:
            user_input = console.input("\n[bold magenta]>[/bold magenta] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[green]再见！[/green]")
                break

            if user_input.startswith('@'):
                # 查询实体相关关系
                entity = user_input[1:].strip()
                relations = ask_about_entity(entity)
                display_results(relations, f"与 {entity} 相关的所有关系")
            else:
                # 自然语言查询
                relations = query_knowledge_graph(user_input)
                display_results(relations, user_input)

        except KeyboardInterrupt:
            console.print("\n[green]再见！[/green]")
            break
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 命令行模式
        if sys.argv[1] == "--help":
            console.print("""
用法:
  python kg-query.py                 # 交互式模式
  python kg-query.py "你的问题"       # 单次查询
  python kg-query.py @实体名          # 查询实体相关关系
            """)
        elif sys.argv[1].startswith('@'):
            entity = sys.argv[1][1:]
            relations = ask_about_entity(entity)
            display_results(relations, f"与 {entity} 相关的所有关系")
        else:
            question = ' '.join(sys.argv[1:])
            relations = query_knowledge_graph(question)
            display_results(relations, question)
    else:
        # 交互式模式
        interactive_mode()
