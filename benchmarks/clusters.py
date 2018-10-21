
clusters = {
    "2x8": [{"cpus": 8}] * 2,
    "4x4": [{"cpus": 4}] * 4,
    "stairs16": [{"cpus": i} for i in range(1, 6)] + [{"cpus": 1}]
}