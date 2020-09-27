# stsearch

Space-time search.


## Install for Development

Create and activate a virtual environment (virtualenv or conda).

```bash
python setup.py --develop
```

## Quick Examples

See [example/README.md](example/README.md)


## stsearch Memo

1. Coordinate system: X, Y is relative between 0 -- 1. T is frame id (0-indexed). We use the same convention as OpenCV:
```
     x_min ------ x_max
y_min
  |
y_max
```

## ffmpeg Memo

1. Segmenting the first 120 seconds of a video and downsample it to 1080p:

```bash
ffmpeg -noaccurate_seek -ss 0 -t 120 -i input.mp4  -vf scale=1080:-1 -movflags frag_keyframe+empty_moov -f mp4  output.mp4
```

## OpenCV Memo

1. When setting `cv2.setNumThreads`, the multi-thread benefit for trackers dimishes very quickly. It's better to use a small number of threads for trackers, and run multiple tracker in parallel using Python threads. On the other hand, we don't want to set `cv2.setNumThreads` too small, because this is a global control that affects other OpenCV functions as well.

Interleaved tracking and LRU decoder time (second) only on a workload:

| cv2.setNumThreads | parallel = 1 | 2  | 4  | 16  |
|-------------------|--------------|----|----|-----|
| -1                | 71           | 46 | 29 | 13  |
| 1                 | 112          | 57 | 48 | 10  |
| 4                 | 78           | 45 | 26 | 9.5 |
| 8                 | 67           | 40 | 23 | 9.1 |
| 16                | 64           | 53 | 22 | 8.9 |

2. `opencv-contribe-python` from PyPI is compiled with pthread rather than OpenMP, so `OMP_NUM_THREADS` seems to have no effect.
