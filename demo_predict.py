from pathlib import Path

from ultralytics import YOLO

# 与仓库内随附的示例图（避免从网络下载图片）
IMAGE = Path(__file__).resolve().parent / "ultralytics" / "assets" / "bus.jpg"

# 本脚本只有在 model.predict() 跑完后才会进入下面的 print；首次运行仍会下载 yolov8n.pt，可能等一会儿。
print("Loading model (first run may download yolov8n.pt)...", flush=True)
model = YOLO("yolov8n.pt")

print(f"Running predict on {IMAGE}...", flush=True)
results = model.predict(
    source=IMAGE,
    conf=0.25,
    verbose=True,
    save=True,
)

for r in results:
    print(r.boxes.data)
    print(r.boxes.xyxy)
    print(r.boxes.xywhn)
    print(r.boxes.conf)
    print(r.boxes.cls)
    # 类别名在 Results 上（COCO 80 类 id→名称），需按每个框的 cls 索引：无 r.boxes.name
    print([r.names[int(c)] for c in r.boxes.cls])