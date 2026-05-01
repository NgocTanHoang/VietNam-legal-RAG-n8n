import os
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download
import pandas as pd
from datasets import load_dataset
import shutil
# import pyarrow.parquet as pq
from bs4 import BeautifulSoup
# import datetime

# Load environment variables from .env file
load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
# login(HF_TOKEN)

# Đường dẫn lưu trữ dữ liệu
save_path = "../data/raw"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def download_legal_subset(subset):
    file_path = os.path.join(save_path, f"legal_{subset}.parquet")

    # Kiểm tra nếu file đã tồn tại thì bỏ qua
    if os.path.exists(file_path):
        print(f"Bộ {subset} đã tồn tại, bỏ qua.")
        return

    print(f"Đang tải bộ {subset}... ")
    try:
        ds = load_dataset(
            "th1nhng0/vietnamese-legal-documents",
            subset,
            split='data'
        )

        df = pd.DataFrame(ds)

        df.to_parquet(file_path, index=False)
        print(f"Đã lưu {subset} thành công! ({len(df)} dòng)")

    except Exception as e:
        print(f"Lỗi khi tải bộ {subset}: {e}")
        
for s in ["metadata", "relationships"]:
    download_legal_subset(s)


dest = os.path.join(save_path, "legal_content.parquet")

if os.path.exists(dest):
    print("File exists: ", dest)
else:
    try:
        path_in_cache = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/content.parquet",
            repo_type="dataset"
        )
        shutil.copy(path_in_cache, dest)
        print(f"Success: {dest}")
    except Exception as e:
        print(f"Error: {e}")

        
#gop bang content với bang metadata     
df_content = pd.read_parquet("../data/raw/legal_content.parquet") 

if not os.path.exists(os.path.join(save_path, "legal_content.parquet")):
    raise ValueError("legal_content.parquet chưa được tạo")

df_meta = pd.read_parquet("../data/raw/legal_metadata.parquet")

df_content['id'] = df_content['id'].astype(str)
df_meta['id'] = df_meta['id'].astype(str)

df_merged = pd.merge(df_content, df_meta, on='id', how='inner')
df_merged.to_parquet("../data/processed/legal_merged.parquet", index=False)



def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%d/%m/%Y", errors='coerce')
    except ValueError:
        return pd.Timestamp.min
    
def clean_html(html_content):
    if not html_content or pd.isna(html_content):
        return ""
    
    soup = BeautifulSoup(html_content, "lxml")
    raw_text = soup.get_text(separator=" ", strip=True)
    clean_text = ' '.join(raw_text.split())
    return clean_text

df_merged['date_dt'] = df_merged['ngay_ban_hanh'].apply(parse_date)
df_sorted = df_merged.sort_values(by='date_dt', ascending=False).head(22000)
df_sorted['content_clean'] = df_sorted['content_html'].apply(clean_html)
df_sorted = df_sorted.fillna('')

df_sorted['ngay_het_hieu_luc'] = df_sorted['ngay_het_hieu_luc'].fillna('31/12/2099')

fill_values = {
    'nganh': 'Không xác định',
    'linh_vuc': 'Không xác định',
    'nguon_thu_nhap': 'van ban phap luat',
    'chuc_danh': 'N/A',
    'nguoi_ky': 'N/A',
    'pham_vi': 'toan quoc',
}

df_sorted = df_sorted.fillna(value=fill_values)

df_sorted = df_sorted[df_sorted['content_clean'].str.strip().astype(bool)]

df_sorted.reset_index(drop=True, inplace=True)
df_sorted.to_parquet("../data/processed/legal_15k.parquet", index=False)

print("Successfully processed and saved legal_15k.parquet with {} documents.".format(len(df_sorted)))