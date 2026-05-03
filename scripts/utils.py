import os
import re
import socket
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from fastembed.text.text_embedding import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
from datetime import datetime
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
    if url: # Ưu tiên URL cho Cloud
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(host=host or "127.0.0.1", port=port or 6333)

def ensure_collection(client, collection_name, vector_size, distance="Cosine", shards=1, replication_factor=1):
    if client.collection_exists(collection_name):
        return
    
    # Với MiniLM (384 dims), cấu hình này là tối ưu
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
        shard_number=shards,
        replication_factor=replication_factor,
    )
    logging.info(f"Đã tạo collection mới: {collection_name} với size {vector_size}")

def load_embedding_model(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir=None, threads=None):
    """Tải model MiniLM."""
    return TextEmbedding(
        model_name=model_name,
        cache_dir=cache_dir or os.getenv('FASTEMBED_CACHE_DIR'),
        threads=threads,
    )

def embed_texts(model, texts, batch_size=64, parallel=1):
    if isinstance(texts, str):
        texts = [texts]
    # Fastembed trả về generator, chuyển thành list list float
    embeddings = list(model.embed(texts, batch_size=batch_size, parallel=parallel))
    return [embedding.tolist() for embedding in embeddings]

def finalize_embedding_text(row):
    """Chuẩn hóa văn bản. Bỏ prefix 'passage:' cho MiniLM"""
    title = row.get('title', "")
    so_hieu = row.get('so_ky_hieu', "")
    loai = row.get('loai_van_ban', "")
    content = row.get('content_clean', "")

    # Tạo chuỗi thông tin cô đọng
    return f"Văn bản: {title}. Số hiệu: {so_hieu}. Loại: {loai}. Nội dung: {content}"

def build_clean_payload(row, rela_dict):
    """Tạo payload và bóc tách thêm năm ban hành để hỗ trợ Agent thống kê"""
    raw_date = row.get('ngay_ban_hanh', '')
    year = None
    
    # Trích xuất năm để hỗ trợ lọc (filtering)
    if raw_date and isinstance(raw_date, str):
        match = re.search(r'\d{4}', raw_date)
        if match:
            year = int(match.group())

    return {
        "id": str(row.get('id', '')),
        "title": row.get('title'),
        "so_ky_hieu": row.get('so_ky_hieu'),
        "ngay_ban_hanh": raw_date,
        "nam_ban_hanh": year, # Thêm trường này để Agent dễ đếm/lọc
        "tinh_trang_hieu_luc": row.get('tinh_trang_hieu_luc'),
        "loai_van_ban": row.get('loai_van_ban'),
        "relationships": rela_dict.get(str(row.get('id', '')), []),
        "content_preview": row.get('content_clean', '')[:1000],
    }

def chunk_text_optimized(row, splitter=None):
    full_text = row.get('text_for_embedding', "")
    base_payload = row.get('payload', {})
    doc_id = base_payload.get('id', 'unknown')

    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150, # Tăng overlap một chút để không mất ngữ cảnh giữa các chunk
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    texts = splitter.split_text(full_text)
    chunks = []

    for i, t in enumerate(texts):
        chunk_payload = base_payload.copy()
        chunk_payload.update({
            "content_preview": t,
            "chunk_id": f"{doc_id}_chunk_{i}",
            "is_chunked": len(texts) > 1,
        })
        # KHÔNG thêm "passage: " ở đây cho model MiniLM
        chunks.append((t, chunk_payload))

    return chunks