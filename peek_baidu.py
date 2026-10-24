# 摸底 Mendeley Baidu 疫情指数数据结构
import pandas as pd, os, glob

BASE = os.path.join(os.path.dirname(__file__), "data", "raw", "baidu_epi")

# wave1: 开放潮窗口（主打）
w1_files = glob.glob(os.path.join(BASE, "**", "*November 26*", "*.xlsx"), recursive=True)
# wave2: 2023 第二波
w2_files = glob.glob(os.path.join(BASE, "**", "*May 6*", "*.xlsx"), recursive=True)
print(f"wave1 文件数: {len(w1_files)}, wave2 文件数: {len(w2_files)}")

def peek(path, tag):
    print(f"\n{'='*70}\n[{tag}] {os.path.basename(path)}")
    xl = pd.ExcelFile(path)
    print(f"sheet 列表: {xl.sheet_names}")
    for sh in xl.sheet_names[:3]:
        df = xl.parse(sh)
        print(f"\n-- sheet [{sh}] shape={df.shape}")
        print(f"列名: {list(df.columns)}")
        print(df.head(4).to_string())
        print("...")
        print(df.tail(3).to_string())

# 全国（wave1）+ 一个省（wave1）
nat1 = [f for f in w1_files if "全国" in f]
peek(nat1[0], "wave1 全国")
anhui1 = [f for f in w1_files if "安徽" in f]
peek(anhui1[0], "wave1 安徽")
# wave2 全国
nat2 = [f for f in w2_files if "全国" in f]
peek(nat2[0], "wave2 全国")
