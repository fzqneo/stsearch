The code here wraps a Docker image built by someone else to expose Yolo inference through HTTP endpoint:
https://github.com/johannestang/yolo_service

It merely converts the result to the format that is used by our Diamond filters.

Limitation:
* No batching
* The returned result doesn't include class IDs