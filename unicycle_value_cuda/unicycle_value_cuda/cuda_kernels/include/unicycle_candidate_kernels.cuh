#pragma once

#include <torch/extension.h>

torch::Tensor build_unicycle_graph_cuda(
    const torch::Tensor& node_states,   // (N,3) float32: x,y,theta_scaled
    const torch::Tensor& controls,      // (A,2) float32: u_lin, u_ang
    const torch::Tensor& bucket_keys,   // (B,2) int32: (bx,by) sorted
    const torch::Tensor& bucket_indices,  // (M) int64: node indices grouped by bucket
    const torch::Tensor& bucket_offsets,  // (B+1) int64
    const torch::Tensor& neighbor_offsets, // (K,2) int32
    const torch::Tensor& origin_xy,        // (2) float32
    double cell_size,
    double dt,
    double rho,
    double angle_scalor,
    int64_t node_start = 0,
    int64_t node_count = -1
);

