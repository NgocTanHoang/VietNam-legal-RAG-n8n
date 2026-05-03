import argparse
import logging
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import gc

# Nạp biến môi trường
load_dotenv()

from utils import (
    build_clean_payload,
    chunk_text_optimized,
    ensure_collection,
    finalize_embedding_text,
    get_qdrant_client,
    embed_texts,
    load_embedding_model,
    generate_uuid,
)

# Cấu hình mặc định mới
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'legal_15k.parquet')
DEFAULT_RELA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'legal_relationships.parquet')

# Đổi model sang MiniLM (384 dims)
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_COLLECTION = "legal_documents_v2" # Nên đổi tên collection vì số chiều vector đã thay đổi
DEFAULT_BATCH_SIZE = 128 # MiniLM nhẹ nên có thể tăng batch size lên lại

def load_data(data_path, rela_path, sample=None):
    logging.info(f"Đang đọc dữ liệu từ: {data_path}")
    df = pd.read_parquet(data_path)
    df['id'] = df['id'].astype(str)
    if sample is not None and sample > 0:
        df = df.head(sample)

    df_rela = pd.read_parquet(rela_path)
    df_rela['doc_id'] = df_rela['doc_id'].astype(str)
    df_rela['other_doc_id'] = df_rela['other_doc_id'].astype(str)

    if sample is not None and sample > 0:
        target_ids = set(df['id'])
        df_rela = df_rela[df_rela['doc_id'].isin(target_ids)]

    rela_dict = (
        df_rela.groupby('doc_id')
        .apply(lambda x: x.to_dict('records'), include_groups=False)
        .to_dict()
    )
    return df, rela_dict

def prepare_data(df, rela_dict):
    # Lưu ý: Model paraphrase thường không bắt buộc prefix "passage: " như E5, 
    # nhưng giữ lại cũng không gây hại nhiều nếu bạn muốn đồng nhất logic.
    df['text_for_embedding'] = df.apply(lambda row: finalize_embedding_text(row), axis=1)
    df['payload'] = df.apply(lambda row: build_clean_payload(row, rela_dict), axis=1)
    return df

def build_chunks(df, splitter=None):
    chunks = []
    for _, row in df.iterrows():
        chunks.extend(chunk_text_optimized(row, splitter))
    return chunks

def upload_chunks(client, collection_name, chunks, model, batch_size=DEFAULT_BATCH_SIZE, parallel=0):
    if not chunks:
        logging.warning('Không có chunk nào để upload.')
        return 0

    # Lấy vector mẫu để xác định size (sẽ là 384)
    sample_vector = embed_texts(model, [chunks[0][0]], batch_size=1, parallel=parallel)[0]
    vector_size = len(sample_vector)
    logging.info(f"Vector size xác định: {vector_size}")
    
    ensure_collection(client, collection_name, vector_size=vector_size, distance='Cosine')

    total_uploaded = 0
    for start in tqdm(range(0, len(chunks), batch_size), desc="Đang đẩy dữ liệu lên Cloud"):
        batch = chunks[start:start + batch_size]
        texts = [text for text, _ in batch]
        payloads = [payload for _, payload in batch]
        # UUID dựa trên chunk_id đảm bảo không trùng lặp khi chạy lại
        ids = [generate_uuid(payload['chunk_id']) for _, payload in batch]

        vectors = embed_texts(model, texts, batch_size=batch_size, parallel=parallel)
        
        client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payloads,
            ids=ids,
            batch_size=batch_size,
            wait=True,
        )
        total_uploaded += len(batch)
        gc.collect() # Giải phóng RAM sau mỗi batch

    return total_uploaded

def parse_args():
    parser = argparse.ArgumentParser(description='Dự án Vietnam Legal RAG - Ingestion Script')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH)
    parser.add_argument('--relations', default=DEFAULT_RELA_PATH)
    parser.add_argument('--collection', default=os.getenv('QDRANT_COLLECTION', DEFAULT_COLLECTION))
    parser.add_argument('--url', default=os.getenv('QDRANT_URL'))
    parser.add_argument('--api-key', default=os.getenv('QDRANT_API_KEY'))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--parallel', type=int, default=0) # Mặc định 0 cho ổn định
    parser.add_argument('--model', default=os.getenv('EMBED_MODEL', DEFAULT_MODEL))
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    if not args.url or not args.api_key:
        logging.error("LỖI: Kiểm tra lại QDRANT_URL/API_KEY trong file .env")
        return

    # Load dữ liệu
    df, rela_dict = load_data(args.data, args.relations, sample=args.sample if args.sample > 0 else None)
    df = prepare_data(df, rela_dict)
    
    # Tạo chunks
    chunks = build_chunks(df)
    
    # Cơ chế bỏ qua bản ghi cũ
    if args.skip > 0:
        logging.info(f"Bỏ qua {args.skip} chunks đầu tiên...")
        chunks = chunks[args.skip:]
    
    gc.collect() 
    
    # Kết nối DB và Model
    client = get_qdrant_client(url=args.url, api_key=args.api_key)
    logging.info(f"Tải model: {args.model}")
    model = load_embedding_model(model_name=args.model)

    # Chạy upload
    total = upload_chunks(
        client,
        collection_name=args.collection,
        chunks=chunks,
        model=model,
        batch_size=args.batch_size,
        parallel=args.parallel,
    )
    
    logging.info(f"Hoàn thành! Đã xử lý tổng cộng {total} points.")

if __name__ == '__main__':
    main()