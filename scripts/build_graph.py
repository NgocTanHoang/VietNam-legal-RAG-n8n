import argparse
import logging
import math
import os
import re
from typing import Dict, Iterable, List

import pandas as pd
from neo4j import GraphDatabase


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "legal_15k.parquet")
DEFAULT_RELA_PATH = os.path.join(BASE_DIR, "data", "raw", "legal_relationships.parquet")
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
DEFAULT_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
DEFAULT_BATCH_SIZE = 500


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def normalize_doc_record(row: pd.Series) -> Dict:
    content_preview = row.get("content_clean") or ""
    return {
        "id": str(row.get("id", "")),
        "title": clean_value(row.get("title")),
        "so_ky_hieu": clean_value(row.get("so_ky_hieu")),
        "ngay_ban_hanh": clean_value(row.get("ngay_ban_hanh")),
        "ngay_co_hieu_luc": clean_value(row.get("ngay_co_hieu_luc")),
        "ngay_het_hieu_luc": clean_value(row.get("ngay_het_hieu_luc")),
        "loai_van_ban": clean_value(row.get("loai_van_ban")),
        "nguon_thu_thap": clean_value(row.get("nguon_thu_thap")),
        "ngay_dang_cong_bao": clean_value(row.get("ngay_dang_cong_bao")),
        "nganh": clean_value(row.get("nganh")),
        "linh_vuc": clean_value(row.get("linh_vuc")),
        "co_quan_ban_hanh": clean_value(row.get("co_quan_ban_hanh")),
        "chuc_danh": clean_value(row.get("chuc_danh")),
        "nguoi_ky": clean_value(row.get("nguoi_ky")),
        "pham_vi": clean_value(row.get("pham_vi")),
        "thong_tin_ap_dung": clean_value(row.get("thong_tin_ap_dung")),
        "tinh_trang_hieu_luc": clean_value(row.get("tinh_trang_hieu_luc")),
        "date_dt": clean_value(row.get("date_dt")),
        "content_preview": content_preview[:4000],
        "content_length": len(content_preview),
        "is_stub": False,
    }


def slugify_relationship(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(name).upper()).strip("_")
    return text or "LIEN_QUAN"


def load_documents(data_path: str, sample: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    df["id"] = df["id"].astype(str)
    if sample and sample > 0:
        df = df.head(sample)
    return df


def load_relationships(rela_path: str, target_ids: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(rela_path)
    df["doc_id"] = df["doc_id"].astype(str)
    df["other_doc_id"] = df["other_doc_id"].astype(str)
    if target_ids:
        df = df[df["doc_id"].isin(target_ids)]
    df = df[df["doc_id"].ne(df["other_doc_id"])]
    df = df.drop_duplicates(subset=["doc_id", "other_doc_id", "relationship"])
    return df


def build_stub_documents(rel_df: pd.DataFrame, known_doc_ids: set[str]) -> List[Dict]:
    referenced_ids = set(rel_df["other_doc_id"].astype(str))
    stub_ids = sorted(referenced_ids - known_doc_ids)
    return [{"id": doc_id, "title": None, "is_stub": True} for doc_id in stub_ids]


def batch_iter(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def create_constraints(driver, database: str) -> None:
    queries = [
        """
        CREATE CONSTRAINT legal_document_id IF NOT EXISTS
        FOR (d:LegalDocument)
        REQUIRE d.id IS UNIQUE
        """,
        """
        CREATE INDEX legal_document_title IF NOT EXISTS
        FOR (d:LegalDocument)
        ON (d.title)
        """,
        """
        CREATE INDEX legal_document_so_ky_hieu IF NOT EXISTS
        FOR (d:LegalDocument)
        ON (d.so_ky_hieu)
        """,
    ]
    with driver.session(database=database) as session:
        for query in queries:
            session.run(query)


def reset_graph(driver, database: str) -> None:
    query = """
    MATCH (d:LegalDocument)
    CALL (d) {
        DETACH DELETE d
    } IN TRANSACTIONS OF 1000 ROWS
    """
    with driver.session(database=database) as session:
        session.run(query).consume()


def upsert_documents(driver, database: str, documents: List[Dict], batch_size: int) -> int:
    query = """
    UNWIND $rows AS row
    MERGE (d:LegalDocument {id: row.id})
    SET d.title = coalesce(row.title, d.title),
        d.so_ky_hieu = coalesce(row.so_ky_hieu, d.so_ky_hieu),
        d.ngay_ban_hanh = coalesce(row.ngay_ban_hanh, d.ngay_ban_hanh),
        d.ngay_co_hieu_luc = coalesce(row.ngay_co_hieu_luc, d.ngay_co_hieu_luc),
        d.ngay_het_hieu_luc = coalesce(row.ngay_het_hieu_luc, d.ngay_het_hieu_luc),
        d.loai_van_ban = coalesce(row.loai_van_ban, d.loai_van_ban),
        d.nguon_thu_thap = coalesce(row.nguon_thu_thap, d.nguon_thu_thap),
        d.ngay_dang_cong_bao = coalesce(row.ngay_dang_cong_bao, d.ngay_dang_cong_bao),
        d.nganh = coalesce(row.nganh, d.nganh),
        d.linh_vuc = coalesce(row.linh_vuc, d.linh_vuc),
        d.co_quan_ban_hanh = coalesce(row.co_quan_ban_hanh, d.co_quan_ban_hanh),
        d.chuc_danh = coalesce(row.chuc_danh, d.chuc_danh),
        d.nguoi_ky = coalesce(row.nguoi_ky, d.nguoi_ky),
        d.pham_vi = coalesce(row.pham_vi, d.pham_vi),
        d.thong_tin_ap_dung = coalesce(row.thong_tin_ap_dung, d.thong_tin_ap_dung),
        d.tinh_trang_hieu_luc = coalesce(row.tinh_trang_hieu_luc, d.tinh_trang_hieu_luc),
        d.date_dt = coalesce(row.date_dt, d.date_dt),
        d.content_preview = coalesce(row.content_preview, d.content_preview),
        d.content_length = coalesce(row.content_length, d.content_length),
        d.is_stub = coalesce(row.is_stub, false)
    """
    total = 0
    with driver.session(database=database) as session:
        for batch in batch_iter(documents, batch_size):
            session.run(query, rows=batch)
            total += len(batch)
            logging.info("Upserted %d/%d document nodes", total, len(documents))
    return total


def upsert_relationships(driver, database: str, relationships: List[Dict], batch_size: int) -> int:
    # Native Cypher - tạo quan hệ với relation_key lưu như property
    # relation_key có thể dùng để filter sau này (ví dụ: WHERE r.relation_key = "THAY_THE")
    query = """
    UNWIND $rows AS row
    MATCH (src:LegalDocument {id: row.doc_id})
    MATCH (dst:LegalDocument {id: row.other_doc_id})
    MERGE (src)-[r:LEGAL_RELATION {source_id: row.doc_id, target_id: row.other_doc_id}]->(dst)
    SET r.relationship = row.relationship,
        r.relation_key = row.relation_key,
        r.relation_group = row.relation_group
    """
    total = 0
    with driver.session(database=database) as session:
        for batch in batch_iter(relationships, batch_size):
            session.run(query, rows=batch)
            total += len(batch)
            logging.info("Upserted %d/%d relationships", total, len(relationships))
    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Build Neo4j graph from Vietnam legal notes")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--relations", default=DEFAULT_RELA_PATH)
    parser.add_argument("--uri", default=DEFAULT_URI, help="Neo4j bolt URI, vd: bolt://localhost:7687")
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xoa toan bo node :LegalDocument va quan he lien quan truoc khi import lai",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="So luong document dau tien de test, 0 la chay toan bo",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    logging.info("Loading documents from %s", args.data)
    doc_df = load_documents(args.data, sample=args.sample if args.sample > 0 else None)
    known_doc_ids = set(doc_df["id"].astype(str))

    logging.info("Loading relationships from %s", args.relations)
    rel_df = load_relationships(args.relations, target_ids=known_doc_ids if args.sample > 0 else None)

    documents = [normalize_doc_record(row) for _, row in doc_df.iterrows()]
    stub_documents = build_stub_documents(rel_df, known_doc_ids)

    relationships = []
    for _, row in rel_df.iterrows():
        relationship = clean_value(row.get("relationship")) or "Lien quan"
        relationships.append(
            {
                "doc_id": str(row["doc_id"]),
                "other_doc_id": str(row["other_doc_id"]),
                "relationship": relationship,
                "relation_key": slugify_relationship(relationship),
                "relation_group": "legal_reference",
            }
        )

    logging.info(
        "Prepared %d full documents, %d stub documents, %d relationships",
        len(documents),
        len(stub_documents),
        len(relationships),
    )

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        driver.verify_connectivity()
        logging.info("Connected to Neo4j at %s", args.uri)

        if args.reset:
            logging.info("Resetting existing LegalDocument graph")
            reset_graph(driver, args.database)

        create_constraints(driver, args.database)

        all_documents = documents + stub_documents
        upsert_documents(driver, args.database, all_documents, args.batch_size)
        upsert_relationships(driver, args.database, relationships, args.batch_size)

        logging.info("Neo4j import completed successfully")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
