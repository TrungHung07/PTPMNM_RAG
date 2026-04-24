# Chunking Strategy Benchmark Results

| Chunk Size | Overlap | Avg Latency (ms) | Hit Rate (@3) | Chunk Count |
|---:|---:|---:|---:|---:|
| 500 | 50 | 35.35 | 0.75 | 6 |
| 500 | 100 | 11.03 | 1.0 | 7 |
| 500 | 200 | 13.45 | 1.0 | 9 |
| 1000 | 50 | 16.51 | 1.0 | 3 |
| 1000 | 100 | 16.51 | 1.0 | 3 |
| 1000 | 200 | 11.22 | 1.0 | 4 |
| 1500 | 50 | 16.05 | 1.0 | 2 |
| 1500 | 100 | 13.14 | 1.0 | 2 |
| 1500 | 200 | 13.25 | 1.0 | 2 |
| 2000 | 50 | 11.32 | 1.0 | 2 |
| 2000 | 100 | 16.97 | 1.0 | 2 |
| 2000 | 200 | 10.83 | 1.0 | 2 |


**Note:** Hit Rate based on keyword matching (threshold >= 50%) in retrieved context @ k=3.