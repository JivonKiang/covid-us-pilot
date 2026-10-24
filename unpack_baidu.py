# 解压 Mendeley Baidu 疫情指数数据集（zip 内文件夹名过长，Expand-Archive 会超 Windows MAX_PATH）
import zipfile, os

SRC = os.path.join(os.path.dirname(__file__), "data", "raw", "baidu_epi_mendeley.zip")
DST = os.path.join(os.path.dirname(__file__), "data", "raw", "baidu_epi")

os.makedirs(DST, exist_ok=True)

# zip 内两个顶层文件夹名都很长 -> 映射成短名
rename_map = {}

with zipfile.ZipFile(SRC) as z:
    for info in z.infolist():
        name = info.filename
        parts = name.split("/")
        if len(parts) > 1 and parts[0] not in rename_map:
            if "May 6" in parts[0]:
                rename_map[parts[0]] = "wave2_2023May-Jul"
            elif "November 26" in parts[0]:
                rename_map[parts[0]] = "wave1_2022Nov-2023Jan"
            else:
                rename_map[parts[0]] = "misc"
        new_name = name
        for old, new in rename_map.items():
            if name.startswith(old):
                new_name = new + name[len(old):]
                break
        target = os.path.join(DST, *new_name.split("/"))
        if info.is_dir() or name.endswith("/"):
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with z.open(info) as fsrc, open(target, "wb") as fdst:
            fdst.write(fsrc.read())
        print(f"{new_name}  {info.file_size} bytes")

print("\n=== 解压完成 ===")
for root, dirs, files in os.walk(DST):
    for f in files:
        p = os.path.join(root, f)
        print(os.path.relpath(p, DST), os.path.getsize(p))
