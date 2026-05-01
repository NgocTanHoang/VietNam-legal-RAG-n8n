import os
import re
import socket
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from fastembed.text.text_embedding import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    """Làm sạch văn bản thô"""
    if not text:
        return ""
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def generate_uuid(raw_id):
    """Tạo UUID đồng nhất từ ID gốc"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))


def check_port_open(host="127.0.0.1", port=6333, timeout=5):
    """Kiểm tra cổng dịch vụ Qdrant trước khi kết nối."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


from qdrant_client import QdrantClient

def get_qdrant_client(url=None, api_key=None, host=None, port=None):
    if url and api_key:
        # Kết nối Cloud
        return QdrantClient(url=url, api_key=api_key)
    # Kết nối Local cũ
    return QdrantClient(host=host, port=port)


def ensure_collection(client, collection_name, vector_size, distance="Cosine", shards=1, replication_factor=1):
    """Tạo collection nếu chưa tồn tại."""
    if client.collection_exists(collection_name):
        return

    vector_params = VectorParams(size=vector_size, distance=distance)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vector_params,
        shard_number=shards,
        replication_factor=replication_factor,
    )


def load_embedding_model(model_name='intfloat/multilingual-e5-large', cache_dir=None, threads=None):
    """Tải model embedding từ fastembed."""
    if cache_dir is None:
        cache_dir = os.getenv('FASTEMBED_CACHE_DIR')

    return TextEmbedding(
        model_name=model_name,
        cache_dir=cache_dir,
        threads=threads,
    )


def embed_texts(model, texts, batch_size=64, parallel=1):
    """Lấy embedding cho một danh sách văn bản."""
    if isinstance(texts, str):
        texts = [texts]

    embeddings = list(model.embed(texts, batch_size=batch_size, parallel=parallel))
    return [embedding.tolist() for embedding in embeddings]


def finalize_embedding_text(row):
    """Chuẩn hóa văn bản trước khi đưa vào model Embedding"""
    title = row.get('title', "Không có tiêu đề")
    so_hieu = row.get('so_ky_hieu', "Không số hiệu")
    loai = row.get('loai_van_ban', "")
    content = row.get('content_clean', "")

    if not content or len(content.split()) < 20:
        return (
            f"Văn bản pháp luật: {title}. Số hiệu: {so_hieu}. "
            f"Loại: {loai}. Nội dung trích yếu đang cập nhật."
        )

    return f"Tiêu đề: {title}. Loại: {loai}. Số hiệu: {so_hieu}. Nội dung: {content}"


def build_clean_payload(row, rela_dict):
    """Tạo dict payload sạch để lưu vào Qdrant"""
    preview = row.get('content_clean', '')[:1000] if row.get('content_clean') else row.get('title', '')

    return {
        "id": str(row.get('id', '')),
        "title": row.get('title'),
        "so_ky_hieu": row.get('so_ky_hieu'),
        "ngay_ban_hanh": row.get('ngay_ban_hanh'),
        "tinh_trang_hieu_luc": row.get('tinh_trang_hieu_luc'),
        "loai_van_ban": row.get('loai_van_ban'),
        "relationships": rela_dict.get(str(row.get('id', '')), []),
        "content_preview": preview,
    }


def get_default_splitter(chunk_size=1000, chunk_overlap=100):
    """Khởi tạo splitter nếu không được truyền vào"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_text_optimized(row, splitter=None):
    """
    Tối ưu việc chia nhỏ văn bản và chuẩn hóa dữ liệu cho Vector Store.
    """
    full_text = row.get('text_for_embedding', "")
    base_payload = row.get('payload', {})
    doc_id = base_payload.get('id', 'unknown')

    if splitter is None:
        splitter = get_default_splitter()

    texts = splitter.split_text(full_text)
    chunks = []

    if len(texts) <= 1:
        final_payload = base_payload.copy()
        final_payload.update({
            "chunk_id": f"{doc_id}_full",
            "is_chunked": False,
            "content_preview": full_text[:1000],
        })
        return [("passage: " + full_text, final_payload)]

    for i, t in enumerate(texts):
        chunk_payload = base_payload.copy()
        chunk_payload.update({
            "content_preview": t,
            "chunk_id": f"{doc_id}_chunk_{i}",
            "is_chunked": True,
        })
        chunks.append(("passage: " + t, chunk_payload))

    return chunks