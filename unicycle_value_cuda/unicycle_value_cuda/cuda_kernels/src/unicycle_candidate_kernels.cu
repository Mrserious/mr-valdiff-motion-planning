#include "unicycle_candidate_kernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

struct CandidateParams {
    const float* node_states;      // (N,3)
    const float2* controls;        // (A,2)
    const int2* bucket_keys;       // (B,2)
    const int64_t* bucket_indices; // (M)
    const int64_t* bucket_offsets; // (B+1)
    const int2* neighbor_offsets;  // (K,2)
    float2 origin;
    float cell_size;
    int64_t num_nodes;
    int64_t num_controls;
    int64_t num_buckets;
    int64_t num_neighbor_offsets;
    float dt;
    float rho;
    float angle_scalor;
    int64_t node_offset;
};

__device__ inline int compare_bucket(const int2& a, int bx, int by) {
    if (a.x == bx) {
        if (a.y == by) {
            return 0;
        }
        return (a.y < by) ? -1 : 1;
    }
    return (a.x < bx) ? -1 : 1;
}

__device__ int64_t find_bucket_index(const int2* keys, int64_t num_buckets, int bx, int by) {
    int64_t left = 0;
    int64_t right = num_buckets;
    while (left < right) {
        int64_t mid = (left + right) / 2;
        int cmp = compare_bucket(keys[mid], bx, by);
        if (cmp == 0) {
            return mid;
        }
        if (cmp < 0) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return -1;
}

__device__ inline float wrap_theta(float theta_scaled, float angle_scalor) {
    const float a = angle_scalor;
    const float p = 2.0f * a;
    // Map to [-a, a) with the same semantics as proj_zero_intval.
    return theta_scaled - floorf((theta_scaled + a) / p) * p;
}

__device__ inline float periodic_abs_diff(float a, float b, float period) {
    float d = fabsf(a - b);
    d = fminf(d, period - d);
    return d;
}

__global__ void preview_candidates_kernel(CandidateParams params, float3* out_candidates) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = params.num_nodes * params.num_controls;
    if (idx >= total) {
        return;
    }
    const int64_t local_node_idx = idx / params.num_controls;
    const int64_t control_idx = idx % params.num_controls;
    const int64_t global_node_idx = local_node_idx + params.node_offset;

    const float* s = params.node_states + global_node_idx * 3;
    const float x = s[0];
    const float y = s[1];
    const float theta_scaled = s[2];

    const float2 ctrl = params.controls[control_idx];
    const float u_lin = ctrl.x;
    const float u_ang = ctrl.y;

    // theta_real = theta_scaled / angle_scalor * pi
    constexpr float kPi = 3.14159265358979323846f;
    const float theta_real = theta_scaled * kPi / params.angle_scalor;
    float3 cand;
    cand.x = x + params.dt * cosf(theta_real) * u_lin;
    cand.y = y + params.dt * sinf(theta_real) * u_lin;
    cand.z = wrap_theta(theta_scaled + params.dt * u_ang, params.angle_scalor);
    out_candidates[idx] = cand;
}

__global__ void count_edges_kernel(
    CandidateParams params,
    const float3* candidate_states,
    int64_t* edge_counts
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = params.num_nodes * params.num_controls;
    if (idx >= total) {
        return;
    }

    const float3 cand = candidate_states[idx];
    const float rho_sq = params.rho * params.rho;
    const float period = 2.0f * params.angle_scalor;

    const float rel_x = (cand.x - params.origin.x) / params.cell_size;
    const float rel_y = (cand.y - params.origin.y) / params.cell_size;
    const int base_x = static_cast<int>(floorf(rel_x));
    const int base_y = static_cast<int>(floorf(rel_y));

    int64_t local_count = 0;
    for (int64_t off_idx = 0; off_idx < params.num_neighbor_offsets; ++off_idx) {
        const int2 offset = params.neighbor_offsets[off_idx];
        const int bucket_x = base_x + offset.x;
        const int bucket_y = base_y + offset.y;
        const int64_t bucket_idx = find_bucket_index(params.bucket_keys, params.num_buckets, bucket_x, bucket_y);
        if (bucket_idx < 0) {
            continue;
        }
        const int64_t start = params.bucket_offsets[bucket_idx];
        const int64_t end = params.bucket_offsets[bucket_idx + 1];
        for (int64_t ptr = start; ptr < end; ++ptr) {
            const int64_t neighbor_idx = params.bucket_indices[ptr];
            const float* ns = params.node_states + neighbor_idx * 3;
            const float dx = ns[0] - cand.x;
            const float dy = ns[1] - cand.y;
            const float dist_xy_sq = dx * dx + dy * dy;
            if (dist_xy_sq > rho_sq) {
                continue;
            }
            const float dtheta = periodic_abs_diff(ns[2], cand.z, period);
            const float dist_sq = dist_xy_sq + dtheta * dtheta;
            if (dist_sq <= rho_sq) {
                ++local_count;
            }
        }
    }
    edge_counts[idx] = local_count;
}

__global__ void write_edges_kernel(
    CandidateParams params,
    const float3* candidate_states,
    const int64_t* edge_offsets,
    int64_t* edge_src,
    int64_t* edge_dst
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = params.num_nodes * params.num_controls;
    if (idx >= total) {
        return;
    }

    const float3 cand = candidate_states[idx];
    const float rho_sq = params.rho * params.rho;
    const float period = 2.0f * params.angle_scalor;

    const int64_t local_node_idx = idx / params.num_controls;
    const int64_t global_node_idx = local_node_idx + params.node_offset;

    const float rel_x = (cand.x - params.origin.x) / params.cell_size;
    const float rel_y = (cand.y - params.origin.y) / params.cell_size;
    const int base_x = static_cast<int>(floorf(rel_x));
    const int base_y = static_cast<int>(floorf(rel_y));

    int64_t write_ptr = edge_offsets[idx];
    for (int64_t off_idx = 0; off_idx < params.num_neighbor_offsets; ++off_idx) {
        const int2 offset = params.neighbor_offsets[off_idx];
        const int bucket_x = base_x + offset.x;
        const int bucket_y = base_y + offset.y;
        const int64_t bucket_idx = find_bucket_index(params.bucket_keys, params.num_buckets, bucket_x, bucket_y);
        if (bucket_idx < 0) {
            continue;
        }
        const int64_t start = params.bucket_offsets[bucket_idx];
        const int64_t end = params.bucket_offsets[bucket_idx + 1];
        for (int64_t ptr = start; ptr < end; ++ptr) {
            const int64_t neighbor_idx = params.bucket_indices[ptr];
            const float* ns = params.node_states + neighbor_idx * 3;
            const float dx = ns[0] - cand.x;
            const float dy = ns[1] - cand.y;
            const float dist_xy_sq = dx * dx + dy * dy;
            if (dist_xy_sq > rho_sq) {
                continue;
            }
            const float dtheta = periodic_abs_diff(ns[2], cand.z, period);
            const float dist_sq = dist_xy_sq + dtheta * dtheta;
            if (dist_sq <= rho_sq) {
                edge_src[write_ptr] = global_node_idx;
                edge_dst[write_ptr] = neighbor_idx;
                ++write_ptr;
            }
        }
    }
}

} // namespace

torch::Tensor build_unicycle_graph_cuda(
    const torch::Tensor& node_states,
    const torch::Tensor& controls,
    const torch::Tensor& bucket_keys,
    const torch::Tensor& bucket_indices,
    const torch::Tensor& bucket_offsets,
    const torch::Tensor& neighbor_offsets,
    const torch::Tensor& origin_xy,
    double cell_size,
    double dt,
    double rho,
    double angle_scalor,
    int64_t node_start,
    int64_t node_count
) {
    TORCH_CHECK(node_states.is_cuda(), "node_states must be CUDA tensor");
    TORCH_CHECK(controls.is_cuda(), "controls must be CUDA tensor");
    TORCH_CHECK(bucket_keys.is_cuda(), "bucket_keys must be CUDA tensor");
    TORCH_CHECK(bucket_indices.is_cuda(), "bucket_indices must be CUDA tensor");
    TORCH_CHECK(bucket_offsets.is_cuda(), "bucket_offsets must be CUDA tensor");
    TORCH_CHECK(neighbor_offsets.is_cuda(), "neighbor_offsets must be CUDA tensor");
    TORCH_CHECK(origin_xy.is_cuda(), "origin_xy must be CUDA tensor");

    TORCH_CHECK(node_states.dim() == 2 && node_states.size(1) == 3, "node_states must be (N,3)");
    TORCH_CHECK(controls.dim() == 2 && controls.size(1) == 2, "controls must be (A,2)");
    TORCH_CHECK(bucket_keys.dim() == 2 && bucket_keys.size(1) == 2, "bucket_keys must be (B,2)");
    TORCH_CHECK(bucket_offsets.dim() == 1, "bucket_offsets must be 1D");
    TORCH_CHECK(bucket_offsets.size(0) == bucket_keys.size(0) + 1, "bucket_offsets must be B+1");
    TORCH_CHECK(neighbor_offsets.dim() == 2 && neighbor_offsets.size(1) == 2, "neighbor_offsets must be (K,2)");
    TORCH_CHECK(origin_xy.numel() == 2, "origin_xy must be (2)");

    TORCH_CHECK(node_states.scalar_type() == torch::kFloat32, "node_states must be float32");
    TORCH_CHECK(controls.scalar_type() == torch::kFloat32, "controls must be float32");
    TORCH_CHECK(bucket_keys.scalar_type() == torch::kInt32, "bucket_keys must be int32");
    TORCH_CHECK(bucket_indices.scalar_type() == torch::kLong, "bucket_indices must be int64");
    TORCH_CHECK(bucket_offsets.scalar_type() == torch::kLong, "bucket_offsets must be int64");
    TORCH_CHECK(neighbor_offsets.scalar_type() == torch::kInt32, "neighbor_offsets must be int32");
    TORCH_CHECK(origin_xy.scalar_type() == torch::kFloat32, "origin_xy must be float32");

    const at::cuda::OptionalCUDAGuard device_guard(node_states.device());

    const int64_t total_nodes = node_states.size(0);
    const int64_t num_controls = controls.size(0);
    TORCH_CHECK(total_nodes > 0, "num_nodes must be > 0");
    TORCH_CHECK(num_controls > 0, "num_controls must be > 0");
    TORCH_CHECK(node_start >= 0, "node_start must be >= 0");
    TORCH_CHECK(node_start < total_nodes, "node_start out of range");

    int64_t chunk_nodes = node_count;
    if (chunk_nodes < 0 || node_start + chunk_nodes > total_nodes) {
        chunk_nodes = total_nodes - node_start;
    }
    TORCH_CHECK(chunk_nodes >= 0, "node_count must be >= 0");
    if (chunk_nodes == 0) {
        auto empty_options = torch::TensorOptions().dtype(torch::kInt64).device(node_states.device());
        return torch::empty({2, 0}, empty_options);
    }

    const torch::Tensor node_states_contig = node_states.contiguous();
    const torch::Tensor controls_contig = controls.contiguous();
    const torch::Tensor bucket_keys_contig = bucket_keys.contiguous();
    const torch::Tensor bucket_indices_contig = bucket_indices.contiguous();
    const torch::Tensor bucket_offsets_contig = bucket_offsets.contiguous();
    const torch::Tensor neighbor_offsets_contig = neighbor_offsets.contiguous();
    const torch::Tensor origin_contig = origin_xy.contiguous();
    const torch::Tensor origin_host = origin_contig.to(torch::kCPU);

    CandidateParams params;
    params.node_states = node_states_contig.data_ptr<float>();
    params.controls = reinterpret_cast<const float2*>(controls_contig.data_ptr<float>());
    params.bucket_keys = reinterpret_cast<const int2*>(bucket_keys_contig.data_ptr<int>());
    params.bucket_indices = bucket_indices_contig.data_ptr<int64_t>();
    params.bucket_offsets = bucket_offsets_contig.data_ptr<int64_t>();
    params.neighbor_offsets = reinterpret_cast<const int2*>(neighbor_offsets_contig.data_ptr<int>());

    const float* origin_ptr = origin_host.data_ptr<float>();
    params.origin = make_float2(origin_ptr[0], origin_ptr[1]);
    params.cell_size = static_cast<float>(cell_size);
    params.num_nodes = chunk_nodes;
    params.num_controls = num_controls;
    params.num_buckets = bucket_keys.size(0);
    params.num_neighbor_offsets = neighbor_offsets.size(0);
    params.dt = static_cast<float>(dt);
    params.rho = static_cast<float>(rho);
    params.angle_scalor = static_cast<float>(angle_scalor);
    params.node_offset = node_start;

    constexpr int kThreads = 256;
    const int64_t total = chunk_nodes * num_controls;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);

    auto candidate_options = node_states_contig.options().dtype(torch::kFloat32);
    torch::Tensor candidate_preview = torch::empty({total, 3}, candidate_options);
    torch::Tensor edge_counts = torch::zeros({total}, node_states_contig.options().dtype(torch::kInt64));

    preview_candidates_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        params,
        reinterpret_cast<float3*>(candidate_preview.data_ptr<float>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    count_edges_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        params,
        reinterpret_cast<const float3*>(candidate_preview.data_ptr<float>()),
        edge_counts.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    torch::Tensor edge_cumsum = edge_counts.cumsum(0);
    int64_t total_edges = 0;
    if (edge_cumsum.numel() > 0) {
        total_edges = edge_cumsum.index({edge_cumsum.size(0) - 1}).item<int64_t>();
    }
    torch::Tensor edge_offsets = torch::zeros_like(edge_counts);
    if (edge_cumsum.numel() > 1) {
        edge_offsets.slice(0, 1, edge_offsets.size(0)).copy_(
            edge_cumsum.slice(0, 0, edge_cumsum.size(0) - 1)
        );
    }

    auto edge_options = torch::TensorOptions().dtype(torch::kInt64).device(node_states.device());
    torch::Tensor edge_src = torch::empty({total_edges}, edge_options);
    torch::Tensor edge_dst = torch::empty({total_edges}, edge_options);

    if (total_edges > 0) {
        write_edges_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            params,
            reinterpret_cast<const float3*>(candidate_preview.data_ptr<float>()),
            edge_offsets.data_ptr<int64_t>(),
            edge_src.data_ptr<int64_t>(),
            edge_dst.data_ptr<int64_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return torch::stack({edge_src, edge_dst}, 0);
}

