import json
import chromadb
from rich.console import Console

console = Console()

def export_to_html(output_file="kg_visualization.html"):
    """将知识图谱导出为 HTML 可视化文件"""

    # 连接 ChromaDB
    db_client = chromadb.PersistentClient(path="./kg_db")
    collection = db_client.get_or_create_collection(name="knowledge_graph")

    # 获取所有数据
    all_data = collection.get()

    if not all_data or not all_data.get('metadatas'):
        console.print("[red]知识图谱为空，请先运行 rag-test.py 构建图谱[/red]")
        return

    console.print(f"[cyan]正在导出 {len(all_data['metadatas'])} 条关系...[/cyan]")

    # 构建节点和边
    nodes = {}
    edges = []

    for metadata in all_data['metadatas']:
        source = metadata['source']
        target = metadata['target']
        relation = metadata['relation']

        if source not in nodes:
            nodes[source] = {"id": source, "label": source}
        if target not in nodes:
            nodes[target] = {"id": target, "label": target}

        edges.append({
            "from": source,
            "to": target,
            "label": relation
        })

    nodes_list = list(nodes.values())
    nodes_json = json.dumps(nodes_list, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)

    # HTML 模板
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识图谱可视化</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
        }
        #header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px 30px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #header h1 {
            font-size: 1.5rem;
            font-weight: 600;
        }
        #header .stats {
            color: #a0a0a0;
            font-size: 0.9rem;
        }
        #network {
            width: 100%;
            height: calc(100vh - 70px);
            border: none;
        }
        #info {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            font-size: 0.85rem;
            max-width: 300px;
            display: none;
            z-index: 1000;
        }
        #info .title {
            font-weight: bold;
            color: #4ecdc4;
            margin-bottom: 8px;
        }
        #info .relation {
            color: #ffe66d;
            margin: 5px 0;
        }
        #controls {
            position: fixed;
            top: 80px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            z-index: 1000;
        }
        #controls button {
            display: block;
            width: 100%;
            padding: 8px 16px;
            margin: 5px 0;
            background: #4ecdc4;
            border: none;
            border-radius: 4px;
            color: #1a1a2e;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
        }
        #controls button:hover {
            background: #45b7aa;
        }
        #search {
            position: fixed;
            top: 80px;
            left: 20px;
            z-index: 1000;
        }
        #search input {
            padding: 10px 15px;
            width: 250px;
            border: none;
            border-radius: 4px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            font-size: 0.9rem;
        }
        #search input::placeholder {
            color: #a0a0a0;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>知识图谱可视化</h1>
        <div class="stats">
            <span id="nodeCount">节点: ''' + str(len(nodes_list)) + '''</span> |
            <span id="edgeCount">关系: ''' + str(len(edges)) + '''</span>
        </div>
    </div>

    <div id="search">
        <input type="text" id="searchInput" placeholder="搜索节点... (回车确认)">
    </div>

    <div id="controls">
        <button onclick="fitAll()">显示全部</button>
        <button onclick="zoomIn()">放大</button>
        <button onclick="zoomOut()">缩小</button>
    </div>

    <div id="network"></div>
    <div id="info">
        <div class="title" id="infoTitle"></div>
        <div class="relation" id="infoRelation"></div>
        <div id="infoDesc"></div>
    </div>

    <script type="text/javascript">
        // 节点数据
        const nodesData = ''' + nodes_json + ''';

        // 边数据
        const edgesData = ''' + edges_json + ''';

        // 创建节点数组
        const nodes = new vis.DataSet(nodesData.map(n => ({
            id: n.id,
            label: n.label,
            title: n.label,
            color: {
                background: '#4ecdc4',
                border: '#3db8ab',
                highlight: { background: '#ffe66d', border: '#ffd93d' }
            }
        })));

        // 创建边数组
        const edges = new vis.DataSet(edgesData.map(e => ({
            from: e.from,
            to: e.to,
            label: e.label,
            arrows: 'to',
            color: { color: '#a0a0a0', highlight: '#ffe66d' },
            font: { color: '#a0a0a0', size: 12, align: 'middle' },
            smooth: { type: 'continuous' }
        })));

        // 创建网络
        const container = document.getElementById('network');
        const data = { nodes: nodes, edges: edges };
        const options = {
            physics: {
                stabilization: { iterations: 100 },
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.1,
                    springLength: 150,
                    springConstant: 0.01
                }
            },
            layout: {
                improvedLayout: true
            },
            nodes: {
                shape: 'dot',
                size: 15,
                font: { color: '#ffffff', size: 14 },
                borderWidth: 2
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            }
        };

        const network = new vis.Network(container, data, options);

        // 显示信息面板
        network.on("click", function(params) {
            const infoPanel = document.getElementById('info');
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                document.getElementById('infoTitle').textContent = '节点: ' + nodeId;
                document.getElementById('infoRelation').textContent = '';
                document.getElementById('infoDesc').textContent = '';
                infoPanel.style.display = 'block';
            } else if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                const edge = edgesData.find(e => e.from === network.body.data.edges.get(edgeId).from &&
                    e.to === network.body.data.edges.get(edgeId).to);
                if (edge) {
                    document.getElementById('infoTitle').textContent = edge.from + ' → ' + edge.to;
                    document.getElementById('infoRelation').textContent = '关系: ' + edge.label;
                    infoPanel.style.display = 'block';
                }
            } else {
                infoPanel.style.display = 'none';
            }
        });

        // 控制函数
        function fitAll() {
            network.fit({ animation: true });
        }

        function zoomIn() {
            const scale = network.getScale() * 1.2;
            network.moveTo({ scale: scale, animation: true });
        }

        function zoomOut() {
            const scale = network.getScale() / 1.2;
            network.moveTo({ scale: scale, animation: true });
        }

        // 搜索功能
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const searchText = this.value.trim();
                if (!searchText) return;

                const matchingNodes = nodesData.filter(n =>
                    n.label.toLowerCase().includes(searchText.toLowerCase())
                );

                if (matchingNodes.length > 0) {
                    const nodeIds = matchingNodes.map(n => n.id);
                    network.selectNodes(nodeIds, true);
                    network.focus(nodeIds[0], {
                        scale: 1.5,
                        animation: true
                    });
                } else {
                    alert('未找到匹配的节点');
                }
            }
        });

        // 初始化时自动适应屏幕
        network.once("stabilizationIterationsDone", function() {
            network.fit({ animation: true });
        });
    </script>
</body>
</html>
'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    console.print(f"[green]导出完成！打开 {output_file} 查看[/green]")
    console.print(f"节点数: {len(nodes_list)}, 边数: {len(edges)}")

if __name__ == "__main__":
    export_to_html()
