from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8-FasterNet.yaml")  # 从头开始构建新模型
# model = YOLO("weights/yolov8n.pt")  # 加载预训练模型（推荐用于训练）


if __name__ == '__main__':
    # Use the model
    results = model.train(data="data/xg.yaml", epochs=100, batch=8)  # 训练模型
