import argparse
import logging
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv  # Thêm thư viện này để nạp .env

# Nạp biến môi trường từ file .env ở thư mục gốc
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'legal_15k.parquet')
DEFAULT_RELA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'legal_relationships.parquet')
DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_COLLECTION = "legal_documents"
DEFAULT_BATCH_SIZE = 32 # Tăng lên để bay nhanh trên Cloud

def load_data(data_path, rela_path, sample=None):
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
    df['text_for_embedding'] = df.apply(finalize_embedding_text, axis=1)
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

    # Lấy vector mẫu để khởi tạo collection
    sample_vector = embed_texts(model, [chunks[0][0]], batch_size=1, parallel=parallel)[0]
    ensure_collection(client, collection_name, vector_size=len(sample_vector), distance='Cosine')

    total_uploaded = 0
    # Dùng tqdm để theo dõi tiến trình trực quan hơn
    for start in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Cloud"):
        batch = chunks[start:start + batch_size]
        texts = [text for text, _ in batch]
        payloads = [payload for _, payload in batch]
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

    return total_uploaded

def parse_args():
    parser = argparse.ArgumentParser(description='Embed dữ liệu và upload CHỈ vào Qdrant Cloud')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH)
    parser.add_argument('--relations', default=DEFAULT_RELA_PATH)
    parser.add_argument('--collection', default=os.getenv('QDRANT_COLLECTION', DEFAULT_COLLECTION))
    # Lấy trực tiếp từ ENV, nếu thiếu sẽ báo lỗi ở main
    parser.add_argument('--url', default=os.getenv('QDRANT_URL'))
    parser.add_argument('--api-key', default=os.getenv('QDRANT_API_KEY'))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    # Sửa lỗi Empty Queue bằng cách mặc định parallel=0 (chạy single process ổn định trên Windows)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--model', default=os.getenv('EMBED_MODEL', DEFAULT_MODEL))
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0, help='Bỏ qua n bản ghi đầu tiên')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    if not args.url or not args.api_key:
        logging.error("LỖI: Thiếu QDRANT_URL hoặc QDRANT_API_KEY. Vui lòng kiểm tra file .env!")
        return

    logging.info('Loading data...')
    df, rela_dict = load_data(args.data, args.relations, sample=args.sample if args.sample > 0 else None)
    df = prepare_data(df, rela_dict)
    
    # 1. Tạo toàn bộ chunks từ dataframe
    chunks = build_chunks(df)
    
    # 2. ĐẶT ĐOẠN CODE SKIP TẠI ĐÂY
    if hasattr(args, 'skip') and args.skip > 0:
        logging.info(f"Đang bỏ qua {args.skip} chunks đầu tiên đã upload...")
        chunks = chunks[args.skip:]
    
    # Giải phóng bớt RAM sau khi đã cắt nhỏ list chunks
    import gc
    gc.collect() 
    
    logging.info('Kết nối tới Qdrant Cloud: %s', args.url)
    client = get_qdrant_client(url=args.url, api_key=args.api_key)

    logging.info('Tải embedding model %s', args.model)
    model = load_embedding_model(model_name=args.model)

    # 3. Tiến hành upload phần còn lại
    total = upload_chunks(
        client,
        collection_name=args.collection,
        chunks=chunks,
        model=model,
        batch_size=args.batch_size,
        parallel=args.parallel,
    )
    
    gc.collect()
    logging.info('Thành công! Đã upload %d điểm mới vào Cloud collection: %s', total, args.collection)

if __name__ == '__main__':
    main()