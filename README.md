# stsearch

Space-time search.


## Install for Development

Create and activate a virtual environment (virtualenv or conda).

```bash
python setup.py --develop
```

## Quick Examples

See [example/README.md](example/README.md)


## OpenCV Memo

1. When setting `cv2.setNumThreads`, the multi-thread benefit for trackers dimishes very quickly. It's better to use a small number of threads for trackers, and run multiple tracker in parallel using Python threads. Although, we don't want to set `cv2.setNumThreads` too small either because this is a global control that affects other OpenCV functions as well.

Interleaved tracking and LRU decoder time (second) only on a workload:

| cv2.setNumThreads | parallel = 1 | 2  | 4  | 16  |
|-------------------|--------------|----|----|-----|
| -1                | 71           | 46 | 29 | 13  |
| 1                 | 112          | 57 | 48 | 10  |
| 4                 | 78           | 45 | 26 | 9.5 |
| 8                 | 67           | 40 | 23 | 9.1 |
| 16                | 64           | 53 | 22 | 8.9 |
