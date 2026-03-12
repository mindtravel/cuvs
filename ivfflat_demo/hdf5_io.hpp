// SPDX-License-Identifier: Apache-2.0
// 读取 ann-benchmarks 生成的 HDF5 文件（train / test / neighbors / distances）

#pragma once

#include <hdf5.h>
#include <cstdint>  // int64_t
#include <stdexcept>
#include <string>
#include <vector>

inline std::string hdf5_err() {
  return "HDF5 error (check file path and format)";
}

struct Hdf5Dataset {
  std::vector<float> train;       // (n_train, dim)
  std::vector<float> test;        // (n_test, dim)
  std::vector<int64_t> neighbors; // (n_test, count) 若存在，与 ann-benchmarks 一致
  std::vector<double> distances_gt;
  size_t n_train{0};
  size_t n_test{0};
  size_t dim{0};
  size_t gt_count{0};
};

inline Hdf5Dataset read_ann_benchmarks_hdf5(const char* path) {
  Hdf5Dataset out;
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    throw std::runtime_error(std::string("cannot open HDF5 file: ") + path);

  auto read_2d_float = [&](const char* name, std::vector<float>& buf, size_t& rows, size_t& cols) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) return;
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2], maxdims[2];
    if (H5Sget_simple_extent_dims(space, dims, maxdims) != 2) {
      H5Sclose(space);
      H5Dclose(dset);
      return;
    }
    rows = dims[0];
    cols = dims[1];
    buf.resize(rows * cols);
    herr_t err = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(space);
    H5Dclose(dset);
    if (err < 0) throw std::runtime_error(hdf5_err());
  };

  auto read_2d_int64 = [&](const char* name, std::vector<int64_t>& buf, size_t& rows, size_t& cols) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) return;
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2], maxdims[2];
    if (H5Sget_simple_extent_dims(space, dims, maxdims) != 2) {
      H5Sclose(space);
      H5Dclose(dset);
      return;
    }
    rows = dims[0];
    cols = dims[1];
    buf.resize(rows * cols);
    herr_t err = H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(space);
    H5Dclose(dset);
    if (err < 0) throw std::runtime_error(hdf5_err());
  };

  auto read_2d_double = [&](const char* name, std::vector<double>& buf, size_t& rows, size_t& cols) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) return;
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2], maxdims[2];
    if (H5Sget_simple_extent_dims(space, dims, maxdims) != 2) {
      H5Sclose(space);
      H5Dclose(dset);
      return;
    }
    rows = dims[0];
    cols = dims[1];
    buf.resize(rows * cols);
    herr_t err = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(space);
    H5Dclose(dset);
    if (err < 0) throw std::runtime_error(hdf5_err());
  };

  read_2d_float("train", out.train, out.n_train, out.dim);
  if (out.train.empty()) throw std::runtime_error("HDF5: missing or empty 'train' dataset");
  read_2d_float("test", out.test, out.n_test, out.dim);
  if (out.test.empty()) throw std::runtime_error("HDF5: missing or empty 'test' dataset");
  size_t nr = 0, nc = 0;
  read_2d_int64("neighbors", out.neighbors, nr, out.gt_count);
  if (!out.neighbors.empty() && (nr != out.n_test || out.gt_count == 0))
    out.neighbors.clear();
  nr = 0;
  nc = 0;
  read_2d_double("distances", out.distances_gt, nr, nc);

  H5Fclose(file);
  return out;
}
