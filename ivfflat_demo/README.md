# IVF-Flat 示例（cuvs）

仅支持 **ann-benchmarks 生成的 HDF5** 作为输入（不再支持 fvecs）。从 HDF5 读入 `train` / `test` / `neighbors`，建 IVF-Flat 索引并检索，用 `neighbors` 算 Recall。

## 依赖

- 已编译 cuvs（指定 `cuvs_DIR` 为 cuvs 的 cpp/build）
- **HDF5 库**（必须）：若 cmake 报错未找到 HDF5，请先安装，例如  
  `conda install -c conda-forge hdf5` 或 `sudo apt install libhdf5-dev`
- 数据：ann-benchmarks 的 HDF5，如 `data/SIFT1M-128-euclidean.hdf5`

## 编译

cuvs 的 **Config 在 cpp/build 里**，不会装进 conda，所以必须用 `cuvs_DIR` 指到 **cuvs 的 cpp/build 目录**（不是 `$CONDA_PREFIX`）。

在已激活编译 cuvs 的 conda 环境下：

```bash
cd /home/diy/lzx/cuvs/ivfflat_demo
conda activate cuvs

mkdir -p build && cd build
# 必须指向含 cuvs-config.cmake 的目录（即 cuvs 的 cpp/build）
cmake -Dcuvs_DIR=/home/diy/lzx/cuvs/cpp/build ..
cmake --build .
```

或用环境变量（同上，指向 cpp/build）：

```bash
export CUVS_BUILD_DIR=/home/diy/lzx/cuvs/cpp/build
cmake -B build && cd build && cmake --build .
```

## 运行

```bash
# 默认使用当前目录下 data/SIFT1M-128-euclidean.hdf5
./ivfflat_demo

# 指定 HDF5 路径（ann-benchmarks 生成或下载的）
./ivfflat_demo data/SIFT1M-128-euclidean.hdf5
./ivfflat_demo /path/to/ann-benchmarks/data/SIFT1M-128-euclidean.hdf5
```

程序会：从 HDF5 读入 `train`（建库）、`test`（查询）、可选 `neighbors`（真值），建 IVF-Flat 索引、做 k=10 检索，并相对 HDF5 中的 `neighbors` 计算 Recall@10。
