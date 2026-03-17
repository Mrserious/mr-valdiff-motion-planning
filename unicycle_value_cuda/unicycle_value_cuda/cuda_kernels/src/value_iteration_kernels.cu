#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <tuple>

namespace {

template <typename scalar_t>
__global__ void value_iteration_warp_kernel(
    int64_t work_nodes,
    int64_t node_offset,
    const int64_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const scalar_t* __restrict__ edge_costs,
    const bool* __restrict__ goal_mask,
    const scalar_t* __restrict__ current_values,
    scalar_t* __restrict__ next_values,
    float* __restrict__ delta_buffer,
    float delta_constant,
    float beta,
    bool has_edge_costs
) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= work_nodes) {
        return;
    }
    const int64_t node_idx = node_offset + warp;
    const bool is_goal = goal_mask[node_idx];
    const float old_value = static_cast<float>(current_values[node_idx]);
    float new_value = old_value;
    const unsigned mask = 0xffffffff;

    if (is_goal) {
        new_value = 0.0f;
    } else {
        const int64_t start = row_offsets[warp];
        const int64_t end = row_offsets[warp + 1];
        float local_best = INFINITY;
        for (int64_t e = start + lane; e < end; e += 32) {
            const int32_t neighbor = col_indices[e];
            const float neighbor_val = static_cast<float>(current_values[neighbor]);
            const float base_cost = has_edge_costs
                ? static_cast<float>(edge_costs[e])
                : delta_constant;
            const float candidate = base_cost + beta * neighbor_val;
            if (candidate < local_best) {
                local_best = candidate;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(mask, local_best, offset);
            if (other < local_best) {
                local_best = other;
            }
        }
        const float best = __shfl_sync(mask, local_best, 0);
        if (lane == 0 && best < INFINITY) {
            new_value = fminf(best, 1.0f);
        }
    }
    if (lane == 0) {
        next_values[node_idx] = static_cast<scalar_t>(new_value);
        delta_buffer[warp] = fabsf(new_value - old_value);
    }
}

}  // namespace

std::tuple<torch::Tensor, int64_t, double> value_iteration_cuda(
    const torch::Tensor& row_offsets,
    const torch::Tensor& col_indices,
    const torch::Tensor& edge_costs,
    const torch::Tensor& goal_mask,
    torch::Tensor values,
    double delta,
    double beta,
    int64_t max_iters,
    double tol,
    bool strict_zero,
    int64_t zero_patience
) {
    TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be CUDA tensor");
    TORCH_CHECK(col_indices.is_cuda(), "col_indices must be CUDA tensor");
    TORCH_CHECK(edge_costs.is_cuda() || edge_costs.numel() == 0, "edge_costs must be CUDA tensor");
    TORCH_CHECK(goal_mask.is_cuda(), "goal_mask must be CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA tensor");
    TORCH_CHECK(row_offsets.dim() == 1, "row_offsets must be 1D");
    TORCH_CHECK(col_indices.dim() == 1, "col_indices must be 1D");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");
    TORCH_CHECK(goal_mask.dim() == 1, "goal_mask must be 1D");
    TORCH_CHECK(max_iters >= 0, "max_iters must be >= 0");
    TORCH_CHECK(row_offsets.scalar_type() == torch::kLong, "row_offsets must use int64");
    TORCH_CHECK(
        col_indices.scalar_type() == torch::kInt || col_indices.scalar_type() == torch::kLong,
        "col_indices must use int32 or int64"
    );
    TORCH_CHECK(goal_mask.scalar_type() == torch::kBool, "goal_mask must be bool tensor");
    TORCH_CHECK(values.is_floating_point(), "values must be floating point tensor");

    const at::cuda::OptionalCUDAGuard device_guard(values.device());

    TORCH_CHECK(row_offsets.device() == values.device(), "row_offsets must be on same device as values");
    TORCH_CHECK(col_indices.device() == values.device(), "col_indices must be on same device as values");
    TORCH_CHECK(goal_mask.device() == values.device(), "goal_mask must be on same device as values");
    if (edge_costs.numel() > 0) {
        TORCH_CHECK(edge_costs.device() == values.device(), "edge_costs must be on same device as values");
    }

    const int64_t row_size = row_offsets.size(0);
    TORCH_CHECK(row_size >= 1, "row_offsets length must be at least 1");
    const int64_t num_nodes = row_size - 1;
    TORCH_CHECK(
        num_nodes <= std::numeric_limits<int32_t>::max(),
        "num_nodes exceeds supported int32 index range"
    );
    TORCH_CHECK(values.size(0) == num_nodes, "values length must equal num_nodes");
    TORCH_CHECK(goal_mask.size(0) == num_nodes, "goal_mask length must equal num_nodes");
    const int64_t num_edges = col_indices.size(0);
    if (edge_costs.numel() > 0) {
        TORCH_CHECK(edge_costs.size(0) == num_edges, "edge_costs must have same length as col_indices");
    }

    if (num_nodes == 0 || max_iters == 0) {
        return std::make_tuple(values.clone(), int64_t{0}, 0.0);
    }

    const torch::Tensor row_offsets_contig = row_offsets.contiguous();
    torch::Tensor col_indices_contig;
    if (col_indices.scalar_type() == torch::kInt) {
        col_indices_contig = col_indices.contiguous();
    } else {
        TORCH_CHECK(
            col_indices.max().item<int64_t>() <= std::numeric_limits<int32_t>::max(),
            "col_indices entries exceed int32 range"
        );
        col_indices_contig = col_indices.to(torch::kInt);
    }
    torch::Tensor values_contig = values.contiguous();
    torch::Tensor current = values_contig.clone();
    torch::Tensor next = torch::empty_like(current);
    torch::Tensor goal_mask_contig = goal_mask.contiguous();
    torch::Tensor edge_costs_contig = edge_costs;
    const bool has_edge_costs = edge_costs.numel() > 0;
    if (has_edge_costs) {
        edge_costs_contig = edge_costs.contiguous();
    }

    auto delta_buffer = torch::empty({num_nodes}, current.options().dtype(torch::kFloat32));

    const int threads = 128;
    const int blocks = static_cast<int>((num_nodes * 32 + threads - 1) / threads);
    const float delta_f = static_cast<float>(delta);
    const float beta_f = static_cast<float>(beta);
    const double tol_d = tol;
    const int64_t patience = std::max<int64_t>(1, zero_patience);

    const int64_t* row_ptr = row_offsets_contig.data_ptr<int64_t>();
    const int32_t* col_ptr = col_indices_contig.data_ptr<int32_t>();
    const bool* goal_ptr = goal_mask_contig.data_ptr<bool>();
    const float tol_f = static_cast<float>(tol);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t zero_hits = 0;
    int64_t iterations_run = 0;
    float last_delta = 0.0f;

    for (int64_t it = 0; it < max_iters; ++it) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            current.scalar_type(),
            "value_iteration_step",
            [&] {
                const scalar_t* edge_ptr = has_edge_costs ? edge_costs_contig.data_ptr<scalar_t>() : nullptr;
                value_iteration_warp_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    num_nodes,
                    /*node_offset=*/0,
                    row_ptr,
                    col_ptr,
                    edge_ptr,
                    goal_ptr,
                    current.data_ptr<scalar_t>(),
                    next.data_ptr<scalar_t>(),
                    delta_buffer.data_ptr<float>(),
                    delta_f,
                    beta_f,
                    has_edge_costs
                );
            }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const float max_delta_val = delta_buffer.max().item<float>();
        last_delta = max_delta_val;
        ++iterations_run;
        std::swap(current, next);

        const bool zero_reached = max_delta_val == 0.0f;
        bool stop = false;
        if (strict_zero) {
            if (zero_reached) {
                ++zero_hits;
                if (zero_hits >= patience) {
                    stop = true;
                }
            } else {
                zero_hits = 0;
            }
        } else if (static_cast<double>(max_delta_val) <= tol_d || max_delta_val <= tol_f) {
            stop = true;
        }

        if (stop) {
            break;
        }
    }

    return std::make_tuple(current, iterations_run, static_cast<double>(last_delta));
}

void value_iteration_chunk_cuda(
    int64_t node_offset,
    const torch::Tensor& row_offsets,
    const torch::Tensor& col_indices,
    const torch::Tensor& goal_mask,
    const torch::Tensor& current_values,
    torch::Tensor next_values,
    double delta,
    double beta,
    torch::Tensor delta_buffer
) {
    TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be CUDA tensor");
    TORCH_CHECK(col_indices.is_cuda(), "col_indices must be CUDA tensor");
    TORCH_CHECK(goal_mask.is_cuda(), "goal_mask must be CUDA tensor");
    TORCH_CHECK(current_values.is_cuda(), "current values must be CUDA tensor");
    TORCH_CHECK(next_values.is_cuda(), "next values must be CUDA tensor");
    TORCH_CHECK(delta_buffer.is_cuda(), "delta buffer must be CUDA tensor");
    TORCH_CHECK(row_offsets.dim() == 1, "row_offsets must be 1D");
    TORCH_CHECK(col_indices.dim() == 1, "col_indices must be 1D");
    TORCH_CHECK(goal_mask.dim() == 1, "goal_mask must be 1D");
    TORCH_CHECK(current_values.dim() == 1, "current values must be 1D");
    TORCH_CHECK(next_values.dim() == 1, "next values must be 1D");
    TORCH_CHECK(delta_buffer.dim() == 1, "delta buffer must be 1D");
    TORCH_CHECK(row_offsets.scalar_type() == torch::kLong, "row_offsets must use int64");
    TORCH_CHECK(
        col_indices.scalar_type() == torch::kInt || col_indices.scalar_type() == torch::kLong,
        "col_indices must use int32 or int64"
    );
    TORCH_CHECK(goal_mask.scalar_type() == torch::kBool, "goal_mask must be bool tensor");
    TORCH_CHECK(current_values.is_floating_point(), "current values must be floating point tensor");
    TORCH_CHECK(next_values.is_floating_point(), "next values must be floating point tensor");

    const at::cuda::OptionalCUDAGuard device_guard(current_values.device());

    TORCH_CHECK(row_offsets.device() == current_values.device(), "row_offsets must be on same device as values");
    TORCH_CHECK(col_indices.device() == current_values.device(), "col_indices must be on same device as values");
    TORCH_CHECK(goal_mask.device() == current_values.device(), "goal_mask must be on same device as values");
    TORCH_CHECK(next_values.device() == current_values.device(), "next_values must be on same device as values");
    TORCH_CHECK(delta_buffer.device() == current_values.device(), "delta_buffer must be on same device as values");

    const int64_t row_size = row_offsets.size(0);
    TORCH_CHECK(row_size >= 1, "row_offsets length must be at least 1");
    const int64_t work_nodes = row_size - 1;
    TORCH_CHECK(work_nodes > 0, "row_offsets must represent at least one node");
    TORCH_CHECK(delta_buffer.size(0) >= work_nodes, "delta_buffer too small");

    const torch::Tensor row_offsets_contig = row_offsets.contiguous();
    torch::Tensor col_indices_contig;
    if (col_indices.scalar_type() == torch::kInt) {
        col_indices_contig = col_indices.contiguous();
    } else {
        TORCH_CHECK(
            col_indices.max().item<int64_t>() <= std::numeric_limits<int32_t>::max(),
            "col_indices entries exceed int32 range"
        );
        col_indices_contig = col_indices.to(torch::kInt);
    }

    const int threads = 128;
    const int blocks = static_cast<int>((work_nodes * 32 + threads - 1) / threads);
    const float delta_f = static_cast<float>(delta);
    const float beta_f = static_cast<float>(beta);

    const int64_t* row_ptr = row_offsets_contig.data_ptr<int64_t>();
    const int32_t* col_ptr = col_indices_contig.data_ptr<int32_t>();
    const bool* goal_ptr = goal_mask.data_ptr<bool>();
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        current_values.scalar_type(),
        "value_iteration_chunk_step",
        [&] {
            value_iteration_warp_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                work_nodes,
                node_offset,
                row_ptr,
                col_ptr,
                /*edge_costs=*/nullptr,
                goal_ptr,
                current_values.data_ptr<scalar_t>(),
                next_values.data_ptr<scalar_t>(),
                delta_buffer.data_ptr<float>(),
                delta_f,
                beta_f,
                /*has_edge_costs=*/false
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

