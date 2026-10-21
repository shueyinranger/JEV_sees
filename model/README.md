```
Training Parameters
    from ultralytics import YOLO

    # Load the model (ensure 'yolov8n.pt' exists or use your custom .pt file)
    model = YOLO('yolo11n.pt')  # or 'yolo11n.pt' if you have a custom model (also can use the trained best.pt to train more epochs)


    # Resume training with new settings
    results = model.train(
        resume = False,
        data="/content/datasets/Vestonly-1/data.yaml",
        epochs=120,  #better to train over 100 epochs
        imgsz=1280,  # In colab, it may have the limitation for training higher pixel
        batch=16,
        device=0,  # 0 for GPU
        workers=2,
        optimizer='AdamW',
        lr0=0.0005,
        weight_decay=0.001,
        patience=8
    )
```