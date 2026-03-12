/*
 * IVF-Flat demo using cuvs (built from source).
 * 数据来源：ann-benchmarks 生成的 HDF5（train / test / neighbors / distances）
 *
 * Build: 见 CMakeLists.txt，需指定 cuvs_DIR 并链接 HDF5。
 * Run:   ./ivfflat_demo [hdf5_path] [batch_size]
 *        batch_size: 参与检索的 query 数量，0 或省略表示用全部 test；例如 1000 表示只用前 1000 条。
 */

#include "hdf5_io.hpp"

#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const char* default_path = "/home/diy/lzx/ann-benchmarks/data/SIFT1M-128-euclidean.hdf5";
  std::string hdf5_path;
  size_t batch_size_req = 0;

  if (argc >= 2 && argv[1][0] != '\0') {
    const std::string a1(argv[1]);
    bool a1_is_number = true;
    for (char c : a1) { if (!std::isdigit(static_cast<unsigned char>(c))) { a1_is_number = false; break; } }
    if (argc == 2 && a1_is_number) {
      batch_size_req = static_cast<size_t>(std::atol(argv[1]));
      hdf5_path = default_path;
    } else {
      hdf5_path = a1;
      if (argc >= 3) batch_size_req = static_cast<size_t>(std::atol(argv[2]));
    }
  } else {
    hdf5_path = default_path;
    if (argc >= 3) batch_size_req = static_cast<size_t>(std::atol(argv[2]));
  }
  if (hdf5_path.empty()) hdf5_path = default_path;

  std::cout << "HDF5: " << hdf5_path << "\n";
  if (batch_size_req > 0)
    std::cout << "Batch size (max queries): " << batch_size_req << "\n";
  std::cout << "Loading ann-benchmarks HDF5 (train / test / neighbors) ...\n";

  Hdf5Dataset ds;
  try {
    ds = read_ann_benchmarks_hdf5(hdf5_path.c_str());
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  const size_t n_base   = ds.n_train;
  const size_t n_test  = ds.n_test;
  const size_t n_query = (batch_size_req > 0 && batch_size_req < n_test) ? batch_size_req : n_test;
  const size_t dim     = ds.dim;

  std::cout << "  train: " << n_base << " x " << dim << "\n";
  std::cout << "  test:  " << n_test << " (using " << n_query << " queries)\n";
  if (!ds.neighbors.empty())
    std::cout << "  neighbors (gt): " << n_test << " x " << ds.gt_count << "\n";

  raft::device_resources dev_resources;
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  const int64_t n_base_i   = static_cast<int64_t>(n_base);
  const int64_t n_query_i = static_cast<int64_t>(n_query);
  const int64_t dim_i     = static_cast<int64_t>(dim);

  auto base_dev  = raft::make_device_matrix<float, int64_t>(dev_resources, n_base_i, dim_i);
  auto query_dev = raft::make_device_matrix<float, int64_t>(dev_resources, n_query_i, dim_i);

  raft::copy(base_dev.data_handle(), ds.train.data(), ds.train.size(),
             raft::resource::get_cuda_stream(dev_resources));
  raft::copy(query_dev.data_handle(), ds.test.data(), n_query * dim,
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  const int64_t n_lists = 1024;
  const int64_t k       = 10;

  std::cout << "Building IVF-Flat index (n_lists=" << n_lists << ") ...\n";
  auto t0 = std::chrono::steady_clock::now();

  cuvs::neighbors::ivf_flat::index_params index_params;
  index_params.n_lists                  = (uint32_t)n_lists;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.metric                   = cuvs::distance::DistanceType::L2Expanded;

  auto index = cuvs::neighbors::ivf_flat::build(
    dev_resources, index_params,
    raft::make_const_mdspan(base_dev.view()));

  raft::resource::sync_stream(dev_resources);
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "  index size=" << index.size() << " build time "
            << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  auto neighbors_dev  = raft::make_device_matrix<int64_t>(dev_resources, n_query_i, k);
  auto distances_dev  = raft::make_device_matrix<float>(dev_resources, n_query_i, k);
  auto neighbors_host = raft::make_host_matrix<int64_t, int64_t>(n_query_i, k);
  auto distances_host = raft::make_host_matrix<float, int64_t>(n_query_i, k);
  rmm::cuda_stream_view stream(raft::resource::get_cuda_stream(dev_resources));

  const std::vector<uint32_t> n_probes_list = {1, 1, 2, 5, 10, 20, 40};
  const bool has_gt = !ds.neighbors.empty() && ds.gt_count >= (size_t)k;
  const int64_t k_gt = has_gt ? std::min((int64_t)ds.gt_count, k) : 0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "n_probes   time(s)      q/s   Recall@" << k_gt << "\n";
  for (uint32_t n_probes : n_probes_list) {
    cuvs::neighbors::ivf_flat::search_params search_params;
    search_params.n_probes = n_probes;

    auto t0 = std::chrono::steady_clock::now();
    cuvs::neighbors::ivf_flat::search(
      dev_resources, search_params, index,
      raft::make_const_mdspan(query_dev.view()),
      neighbors_dev.view(), distances_dev.view());
    raft::resource::sync_stream(dev_resources);
    auto t1 = std::chrono::steady_clock::now();
    double search_s = std::chrono::duration<double>(t1 - t0).count();

    raft::copy(neighbors_host.data_handle(), neighbors_dev.data_handle(), neighbors_dev.size(), stream);
    raft::copy(distances_host.data_handle(), distances_dev.data_handle(), distances_dev.size(), stream);
    raft::resource::sync_stream(dev_resources);

    double recall = 0.0;
    if (has_gt && k_gt > 0) {
      int64_t match = 0;
      for (int64_t i = 0; i < n_query_i; i++)
        for (int64_t j = 0; j < k_gt; j++)
          if (neighbors_host(i, j) == (int64_t)ds.neighbors[i * ds.gt_count + j])
            match++;
      recall = match / (double)(n_query_i * k_gt);
    }
    std::cout << std::setw(8) << n_probes
              << std::setw(12) << search_s
              << std::setw(12) << std::setprecision(1) << (static_cast<double>(n_query) / search_s)
              << std::setw(12) << std::setprecision(4) << recall << "  (batch=" << n_query << ")\n";
  }

  std::cout << "First query (n_probes=10) top-" << k << " indices: ";
  {
    cuvs::neighbors::ivf_flat::search_params sp;
    sp.n_probes = 10;
    cuvs::neighbors::ivf_flat::search(dev_resources, sp, index,
      raft::make_const_mdspan(query_dev.view()), neighbors_dev.view(), distances_dev.view());
    raft::resource::sync_stream(dev_resources);
    raft::copy(neighbors_host.data_handle(), neighbors_dev.data_handle(), neighbors_dev.size(), stream);
    raft::resource::sync_stream(dev_resources);
  }
  for (int64_t j = 0; j < k; j++)
    std::cout << neighbors_host(0, j) << (j == k - 1 ? "\n" : ", ");

  std::cout << "Done.\n";
  return 0;
}
