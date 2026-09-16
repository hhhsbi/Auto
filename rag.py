# rag.py
# =============================================================================
# 模块导入
# =============================================================================
import hashlib  # 计算文件内容 sha256 作为 doc_id
import os  # 用于文件和目录路径操作
from pathlib import Path

from tracing import traceable, add_run_metadata  # LangSmith 监控（langsmith 未安装时自动降级为空实现）
# 阶段 4 两级缓存：retrieve 结果按 query+role+top_k+model_version 缓存（LRU + Redis）；
# cache 模块加载时会自检 Redis，未启则降级为只用一级 LRU
from cache import cache, EMBEDDING_MODEL_VERSION as _CACHE_EMBEDDING_VERSION

# embedding 模型版本（与 cache.py 里 EMBEDDING_MODEL_VERSION 同款值；
# 改模型时改这两处，cache key 里的 model_version 段自然变，全量缓存失效）
EMBEDDING_MODEL_VERSION = "paraphrase-multilingual-MiniLM-L12-v2"
assert EMBEDDING_MODEL_VERSION == _CACHE_EMBEDDING_VERSION, "rag.EMBEDDING_MODEL_VERSION 与 cache.py 不一致，缓存会失效"

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

# 哨兵标签：Chroma 列表元数据不允许空列表，用哨兵值表达"公开/占位"
PUBLIC_TAG = "__all__"  # 公开：任何人可见
NONE_TAG = "__none__"   # 占位：永不匹配任何真实角色/部门


def parse_tags(tags):
    """把 str/list/None 统一解析成小写去重的标签列表。"""
    if not tags:
        return []
    if isinstance(tags, str):
        tags = tags.split(",")
    seen = []
    for t in tags:
        t = str(t).strip().lower()
        if t and t not in seen:
            seen.append(t)
    return seen


def tags_to_metadata(allowed_roles, allowed_departments):
    """
    把权限标签转成 Chroma 列表元数据（Chroma 1.x 支持列表 + $contains 成员匹配，
    但要求列表非空，所以用哨兵值占位）。

    规则：
      两边都空                -> 公开文档：(["__all__"], ["__all__"])
      只限角色，部门不限制     -> (["hr",...], ["__none__"])
      只限部门，角色不限制     -> (["__none__"], ["tech",...])
    """
    roles, depts = parse_tags(allowed_roles), parse_tags(allowed_departments)
    if not roles and not depts:
        return [PUBLIC_TAG], [PUBLIC_TAG]
    return roles or [NONE_TAG], depts or [NONE_TAG]


def build_permission_where(role=None, department=None):
    """
    构造 Chroma where 权限过滤条件（配合列表元数据 + $contains 成员匹配）。

    可见规则：公开文档 OR 角色被授权 OR 部门被授权；admin 不过滤（全库可见）；
    未认证上下文（role 为 None）只能看公开文档（安全默认：不给权限就只看公开的）。

    返回 None 表示不过滤（admin）。
    """
    if role == "admin":
        return None
    clauses = [{"allowed_roles": {"$contains": PUBLIC_TAG}}]
    if role:
        clauses.append({"allowed_roles": {"$contains": role}})
    if department:
        clauses.append({"allowed_departments": {"$contains": department}})
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def backfill_permission_tags():
    """
    存量数据迁移（幂等）：把旧格式的权限字段统一成列表元数据。

    覆盖三种旧格式：字段缺失、空串 ""（=公开）、逗号围栏串 ",hr,"（阶段3早期格式，
    该格式在 Chroma 1.5.9 里 $contains 不生效，必须迁移成列表）。
    """
    data = collection.get()
    ids, metas = data.get("ids", []), data.get("metadatas", [])
    need_ids, need_metas = [], []
    for cid, meta in zip(ids, metas):
        meta = meta or {}
        if isinstance(meta.get("allowed_roles"), list) and isinstance(meta.get("allowed_departments"), list):
            continue  # 已是新格式
        roles_meta, depts_meta = tags_to_metadata(
            meta.get("allowed_roles"), meta.get("allowed_departments")
        )
        need_ids.append(cid)
        need_metas.append({**meta, "allowed_roles": roles_meta, "allowed_departments": depts_meta})
    if need_ids:
        collection.update(ids=need_ids, metadatas=need_metas)
        print(f"[rag] 已为 {len(need_ids)} 个存量块迁移权限标签为列表格式（默认公开）")


# 存量数据迁移（幂等）：在函数定义后执行
backfill_permission_tags()


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    固定长度滑动窗口切块（阶段 1 升级，替代旧的按空行切块）。

    策略：按 chunk_size 个字符开窗，窗口间回退 chunk_overlap 个字符形成重叠，
    保证跨窗口的句子不会因为切断而检索不到。在窗口尾部优先回退到段落（空行）
    或句号边界，避免把句子拦腰截断；单段超过 chunk_size 时才硬切。

    参数:
        text (str): 原始文档内容
        chunk_size (int): 每块最大字符数
        chunk_overlap (int): 相邻块之间的重叠字符数

    返回:
        list[str]: 切块列表（不含空块）
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size，否则窗口无法前进")
    text = text.strip()
    n = len(text)
    if n == 0:
        return []

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # 只在还有后续内容时找边界：优先段落空行，其次句号/换行
            seg = text[start:end]
            cut = max(seg.rfind('\n\n'), seg.rfind('。'), seg.rfind('\n'))
            # 回退太多（切点在窗口前半段）会导致块过小，宁可直接硬切
            if cut > chunk_size // 2:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # 下一窗口向回借 overlap 个字符形成重叠；max 保证至少前进 1 字符防死循环
        start = max(end - chunk_overlap, start + 1)
    return chunks


def index_chunks(doc_id, source, chunks, batch_size=32, progress_cb=None,
                 allowed_roles="", allowed_departments=""):
    """
    把切块分批向量化入库（供后台异步入库管道调用）。

    分批是为了大批量文件入库时能通过 progress_cb 实时汇报进度，
    而不是把所有块一次性 embedding 完才返回。

    参数:
        doc_id (str): 文档唯一 ID（内容 sha256），写入每块的元数据
        source (str): 来源文件名
        chunks (list[str]): chunk_text 的输出
        batch_size (int): 每批入库的块数
        progress_cb (callable): 进度回调 progress_cb(done, total)，可为 None
        allowed_roles (str|list): 可见角色，空=不限（配合 departments 判定是否公开）
        allowed_departments (str|list): 可见部门，空=不限

    返回:
        int: 入库的总块数
    """
    roles_meta, depts_meta = tags_to_metadata(allowed_roles, allowed_departments)
    total = len(chunks)
    done = 0
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=batch,
            ids=[f"{doc_id}_{i + j}" for j in range(len(batch))],
            metadatas=[
                {
                    "source": source,
                    "chunk_index": i + j,
                    "doc_id": doc_id,
                    "allowed_roles": roles_meta,
                    "allowed_departments": depts_meta,
                }
                for j in range(len(batch))
            ],
        )
        done += len(batch)
        if progress_cb:
            progress_cb(done, total)
    return total


def delete_doc(doc_id):
    """按 doc_id 删除一个文档的全部块（重传/重试时先清理旧块，保证幂等）。"""
    collection.delete(where={"doc_id": doc_id})


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
        # 按固定长度滑动窗口切块（含重叠）
        chunks = chunk_text(text)
        if not chunks:
            continue  # 空文件跳过

        # 文件内容 sha256 作为 doc_id，写入每块元数据，支持按文档维度删除/失效
        doc_id = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # 为每个块生成唯一 ID（doc_id_块索引）
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        # 为每个块添加元数据：来源文件名、块序号、所属文档 ID、权限标签（docs/ 默认全公开）
        metadatas = [
            {"source": filename, "chunk_index": i, "doc_id": doc_id,
             "allowed_roles": [PUBLIC_TAG], "allowed_departments": [PUBLIC_TAG]}
            for i in range(len(chunks))
        ]

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

@traceable(name="retrieve", run_type="retriever")
def retrieve(query, top_k=3, role=None, department=None):
    """
    根据用户问题检索最相关的文档块（带 RBAC 权限过滤）。

    缓存（阶段 4）：先查两级缓存（LRU + Redis），命中直接返；miss 才查 Chroma，
    查完写回两级缓存。缓存 key 含 role / top_k / model_version，权限或模型变自然失效；
    文件更新走 invalidate_doc 按反向索引失效。

    参数:
        query (str): 用户输入的问题
        top_k (int): 返回最相关的前 k 个块，默认 3
        role (str): 当前用户角色；None=未认证上下文，只能看公开文档；"admin" 不过滤
        department (str): 当前用户部门；配合 allowed_departments 标签过滤

    返回:
        list[dict]: 每个字典包含 'text'（块内容）、'metadata'（元数据）、'distance'（相似度距离）
    """
    # 1) 先查两级缓存：命中直接返，跳过向量化
    cached, hit_source = cache.get_retrieve(query, top_k, role, department)
    if cached is not None:
        add_run_metadata({
            "召回条数": len(cached),
            "缓存命中": hit_source,  # "lru" / "redis"
        })
        return cached

    # 2) 缓存 miss → 走 Chroma 向量检索
    # collection.query 执行向量检索：先将 query 转成向量，然后与库中所有向量比较，
    # 按余弦距离升序返回最相似的 top_k 个结果。where 先做权限过滤再排序。
    where = build_permission_where(role=role, department=department)
    query_kwargs = {"query_texts": [query], "n_results": top_k}
    if where is not None:
        query_kwargs["where"] = where
    results = collection.query(**query_kwargs)

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
    # 召回条数与缓存 miss 写入 LangSmith 元数据
    add_run_metadata({"召回条数": len(retrieved), "缓存命中": None})
    # 3) 写回两级缓存（set_retrieve 内部会维护 doc_id 反向索引，供失效用）
    cache.set_retrieve(query, top_k, role, department, retrieved)
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