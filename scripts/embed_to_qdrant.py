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

# Cấu hình mặc định
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'legal_15k.parquet')
DEFAULT_RELA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'legal_relationships.parquet')

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_COLLECTION = "legal_documents_v2"
DEFAULT_BATCH_SIZE = 128 

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
    # Tạo cột 'text_for_embedding' và 'payload' mà utils.chunk_text_optimized yêu cầu
    df['text_for_embedding'] = df.apply(lambda row: finalize_embedding_text(row), axis=1)
    df['payload'] = df.apply(lambda row: build_clean_payload(row, rela_dict), axis=1)
    return df

def build_chunks(df, splitter=None):
    chunks = []
    # Thêm tqdm ở đây để bạn theo dõi tiến trình tạo chunk (tránh cảm giác bị treo)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Đang tạo chunks"):
        chunks.extend(chunk_text_optimized(row, splitter))
    return chunks

def upload_chunks(client, collection_name, chunks, model, batch_size=DEFAULT_BATCH_SIZE, parallel=0):
    if not chunks:
        logging.warning('Không có chunk nào để upload.')
        return 0

    sample_vector = embed_texts(model, [chunks[0][0]], batch_size=1, parallel=parallel)[0]
    vector_size = len(sample_vector)
    logging.info(f"Vector size xác định: {vector_size}")
    
    ensure_collection(client, collection_name, vector_size=vector_size, distance='Cosine')

    total_uploaded = 0
    for start in tqdm(range(0, len(chunks), batch_size), desc="Đang đẩy dữ liệu lên Qdrant"):
        batch = chunks[start:start + batch_size]
        texts = [text for text, _ in batch]
        payloads = [payload for _, payload in batch]
        ids = [generate_uuid(payload['chunk_id']) for _, payload in batch]

        vectors = embed_texts(model, texts, batch_size=batch_size, parallel=parallel)
        
        # Lưu ý: Sử dụng upload_collection hoặc upsert tùy theo phiên bản qdrant-client
        client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payloads,
            ids=ids,
            batch_size=batch_size,
            wait=True,
        )
        total_uploaded += len(batch)
        gc.collect() 

    return total_uploaded

def parse_args():
    parser = argparse.ArgumentParser(description='Vietnam Legal RAG Ingestion')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH)
    parser.add_argument('--relations', default=DEFAULT_RELA_PATH)
    parser.add_argument('--collection', default=os.getenv('QDRANT_COLLECTION', DEFAULT_COLLECTION))
    parser.add_argument('--url', default=os.getenv('QDRANT_URL'))
    parser.add_argument('--api-key', default=os.getenv('QDRANT_API_KEY'))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--model', default=os.getenv('EMBED_MODEL', DEFAULT_MODEL))
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    if not args.url or not args.api_key:
        logging.error("LỖI: Thiếu QDRANT_URL hoặc QDRANT_API_KEY")
        return

    # 1. Khởi tạo Client và Model
    client = get_qdrant_client(args.url, args.api_key)
    model = load_embedding_model(args.model)

    # 2. Xử lý dữ liệu
    df, rela_dict = load_data(args.data, args.relations, sample=args.sample if args.sample > 0 else None)
    df = prepare_data(df, rela_dict)
    
    # 3. Tạo toàn bộ chunks
    chunks = build_chunks(df)
    total_chunks = len(chunks)
    logging.info(f"Tổng số chunks đã tạo: {total_chunks}")
    
    # 4. Cơ chế SKIP
    if args.skip > 0:
        if args.skip >= total_chunks:
            logging.warning(f"Số lượng skip ({args.skip}) lớn hơn tổng số chunk. Thoát.")
            return
        logging.info(f"Bỏ qua {args.skip} chunks đầu tiên...")
        chunks = chunks[args.skip:]
    
    logging.info(f"Số lượng chunk thực tế sẽ upload: {len(chunks)}")
    gc.collect() 

    # 5. GỌI HÀM UPLOAD (Phần quan trọng nhất bị thiếu)
    total_up = upload_chunks(
        client=client,
        collection_name=args.collection,
        chunks=chunks,
        model=model,
        batch_size=args.batch_size,
        parallel=args.parallel
    )
    
    logging.info(f"Thành công! Đã upload {total_up} chunks.")

if __name__ == "__main__":
    main()