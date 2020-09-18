FROM johannestang/yolo_service:1.0_cuda10.0-yolov3_coco

COPY ["wrap_app.py", "wrap_entrypoint.sh", "/app/"]
CMD ["./wrap_entrypoint.sh"]