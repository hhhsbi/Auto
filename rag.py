# rag.py
# =============================================================================
# 模块导入
# =============================================================================
import os  # 用于文件和目录路径操作
from pathlib import Path

# 项目根目录（绝对路径），保证从任何工作目录运行都能正确定位 docs/chroma_data。
BASE_DIR = Path(__file__).resolve().parent

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 添加这一行
import chromadb  # Chroma 向量数据库，用于存储和检索文档向量
from chromadb import Documents, EmbeddingFunction, Embeddings  # 自定义 embedding 函数的基类与类型注解
from sentence_transformers import SentenceTransformer  # 加载本地 embedding 模型
from openai import OpenAI  # OpenAI 兼容客户端，用于调用 DeepSeek API
from dotenv import load_dotenv  # 从 .env 文件加载环境变量

# =============================================================================
# 1. 配置部分
# =============================================================================

# 加载 .env 文件中的环境变量（如 DEEPSEEK_API_KEY）
load_dotenv()

# 从环境变量获取 DeepSeek API Key，若不存在则报错
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

# 初始化 DeepSeek 客户端，使用 OpenAI 兼容接口
# base_url 指向 DeepSeek 的 API 端点
client_openai = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# 初始化一个多语言 Sentence-BERT 模型，用于将文本转换成向量（embedding）
# 该模型支持中文，轻量且适合本地运行
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# 自定义一个 embedding 函数类，供 Chroma 调用
# Chroma 在添加文档或查询时，会调用此类的 __call__ 方法将文本转成向量
class MyEmbeddingFunction(EmbeddingFunction):
    # 继承 Chroma 官方的 EmbeddingFunction 基类（新版推荐做法）。
    # 这样 name() 方法会自动提供，且 __call__ 的参数名必须叫 input。

    def __init__(self):
        # 本类没有自己的状态要初始化，留空即可。
        # 注意：不要调用 super().__init__()——基类的 __init__ 本身只会发"请实现 __init__"的警告，
        # 实际什么也不做；定义这个空 __init__ 就能消除那个 DeprecationWarning。
        pass

    def __call__(self, input: Documents) -> Embeddings:
        # 参数名必须叫 input（不能叫 texts），否则 Chroma 校验签名时报错。
        return embed_model.encode(input).tolist()


# 初始化 Chroma 持久化客户端，数据存储在 ./chroma_data 目录
# 这样即使程序重启，之前存入的文档也不会丢失
chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_data"))

# 获取或创建一个名为 "docs_collection" 的集合（类似数据库中的表）
# 并指定我们自定义的 embedding 函数，确保向量化方式一致
collection = chroma_client.get_or_create_collection(
    name="docs_collection",
    embedding_function=MyEmbeddingFunction()
)


# =============================================================================
# 2. 文档切块与入库函数
# =============================================================================

def chunk_by_paragraph(text):
    """
    按空行（段落）切分文本，保留段落完整性。

    参数:
        text (str): 原始文档内容

    返回:
        list[str]: 段落列表，每个段落前后空白被去除，空段落被过滤
    """
    # 使用 split('\n\n') 按两个换行符切分，即空行分隔
    # 然后 strip() 去除首尾空格，只保留非空段落
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


def index_documents(docs_dir):
    """
    扫描指定目录下的 .txt 和 .md 文件，切块后存入 Chroma 集合。
    如果集合中已有数据，则跳过索引（避免重复添加）。

    参数:
        docs_dir (str): 存放文档的目录路径
    """
    # 若集合中已有文档块，则提示并跳过，防止重复索引
    if collection.count() > 0:
        print(f"知识库已有 {collection.count()} 个块，跳过索引（如需重新索引请删除 ./chroma_data 文件夹）。")
        return

    print("开始索引文档...")
    # 遍历目录下的所有文件
    for filename in os.listdir(docs_dir):
        # 只处理 .txt 和 .md 文件
        if not filename.endswith(('.txt', '.md')):
            continue
        filepath = os.path.join(docs_dir, filename)
        # 以 UTF-8 编码读取文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        # 按段落切分
        chunks = chunk_by_paragraph(text)
        if not chunks:
            continue  # 空文件跳过

        # 为每个块生成唯一 ID（文件名_块索引）
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        # 为每个块添加元数据：来源文件名和块序号
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        # 将文档块、元数据和 ID 一起添加到 Chroma 集合
        # Chroma 内部会自动调用 MyEmbeddingFunction 将 documents 转为向量
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  已索引 {filename}，共 {len(chunks)} 个块")
    print(f"索引完成，总块数：{collection.count()}")


# =============================================================================
# 3. 检索与 Prompt 构建函数
# =============================================================================

def retrieve(query, top_k=3):
    """
    根据用户问题检索最相关的文档块。

    参数:
        query (str): 用户输入的问题
        top_k (int): 返回最相关的前 k 个块，默认 3

    返回:
        list[dict]: 每个字典包含 'text'（块内容）、'metadata'（元数据）、'distance'（相似度距离）
    """
    # collection.query 执行向量检索：先将 query 转成向量，然后与库中所有向量比较，
    # 按余弦距离升序返回最相似的 top_k 个结果。
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    # results 是一个字典，包含 'documents'、'metadatas'、'distances'（距离）等
    # 注意：返回的值都是列表的列表（因为可以一次查询多个问题，我们只传一个）
    docs = results['documents'][0]  # 文本内容列表
    metas = results['metadatas'][0]  # 元数据列表
    distances = results['distances'][0] if 'distances' in results else None  # 距离列表

    # 组合成更易用的结构
    retrieved = []
    for i in range(len(docs)):
        retrieved.append({
            "text": docs[i],
            "metadata": metas[i],
            "distance": distances[i] if distances else None
        })
    return retrieved


def build_rag_prompt(query, retrieved):
    """
    根据检索到的文档块构造一个严格的系统提示词（prompt），
    强制 LLM 只基于这些参考资料回答，并注明出处。

    参数:
        query (str): 用户问题
        retrieved (list[dict]): 由 retrieve 函数返回的检索结果

    返回:
        str: 完整的用户消息内容，可直接发送给 LLM
    """
    refs = ""
    # 枚举检索结果，为每个块编号并附上来源文件名
    for i, item in enumerate(retrieved):
        src = item['metadata'].get('source', '未知来源')
        refs += f"[{i + 1}] 来自 {src}\n{item['text']}\n\n"

    # 构建 prompt，明确要求 LLM 只使用参考资料，并标注引用
    prompt = f"""你是一个基于事实的回答助手。请只根据下面的参考资料回答问题。
如果参考资料中没有相关信息，请回答“根据现有资料，无法确认”。

=== 参考资料 ===
{refs}
=== 问题 ===
{query}
=== 回答要求 ===
- 回答必须引用参考资料中的具体信息，并在每句话后面标注来源编号（例如 [1]）。
- 不要添加任何参考资料中没有的信息。
- 如果信息不充分，明确说明。

现在请回答：
"""
    return prompt


def ask_rag(query):
    """
    RAG 完整流程：检索 -> 构造 prompt -> 调用 LLM 生成回答。

    参数:
        query (str): 用户问题

    返回:
        tuple: (answer, retrieved)
            - answer (str): LLM 生成的回答文本
            - retrieved (list[dict]): 检索到的文档块列表（用于展示引用）
    """
    # 1. 检索最相关的 3 个块
    retrieved = retrieve(query, top_k=3)
    if not retrieved:
        return "未找到任何相关文档。", []

    # 2. 根据检索结果构建 prompt
    prompt = build_rag_prompt(query, retrieved)

    # 3. 调用 DeepSeek API 生成回答
    # temperature 设为 0.2 较低，使输出更确定、更忠于提供的资料
    response = client_openai.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    answer = response.choices[0].message.content
    return answer, retrieved


# =============================================================================
# 4. 交互式命令行主程序
# =============================================================================

def main():
    """
    主函数：
    1. 确保 ./docs 目录存在，并索引其中的文档（如果尚未索引）。
    2. 进入循环，接受用户输入问题，调用 RAG 流程并打印回答及引用来源。
    """
    docs_dir = str(BASE_DIR / "docs")
    # 如果 docs 目录不存在，则创建并提示用户放入文档
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"请把文档放入 {docs_dir} 目录，然后重新运行。")
        return

    # 执行索引（若已索引则跳过）
    index_documents(docs_dir)

    print("\n===== RAG 问答系统已启动（输入 exit 退出）=====")
    # 交互循环
    while True:
        query = input("\n请输入问题：").strip()
        if query.lower() == 'exit':
            break
        if not query:
            continue  # 空输入则重新提示

        try:
            # 调用 RAG 流程
            answer, retrieved = ask_rag(query)
            print("\n--- 回答 ---")
            print(answer)
            print("\n--- 参考来源 ---")
            # 打印每个引用块的来源文件名和块索引
            for i, item in enumerate(retrieved):
                src = item['metadata'].get('source', '未知')
                print(f"[{i + 1}] {src} (片段 {item['metadata'].get('chunk_index', '')})")
        except Exception as e:
            print(f"发生错误：{e}")


# 如果直接运行此脚本，则执行 main()
if __name__ == "__main__":
    main()