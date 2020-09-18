REGISTRY ?= registry.cmusatyalab.org
IMAGEID ?= $(REGISTRY)/zf/diamond-public-registry/filter/yolo_wrap

default:
	docker build --pull -t $(IMAGEID) .
	docker push $(IMAGEID)

detection-up:
	docker run --gpus all --name yolo_service --rm -t -d -p 5001:5000 $(IMAGEID)

