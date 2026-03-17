#include <torch/extension.h>

namespace py = pybind11;

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
);

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
);

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "build_graph",
        &build_unicycle_graph_cuda,
        "Build unicycle graph (CUDA)",
        py::arg("node_states"),
        py::arg("controls"),
        py::arg("bucket_keys"),
        py::arg("bucket_indices"),
        py::arg("bucket_offsets"),
        py::arg("neighbor_offsets"),
        py::arg("origin_xy"),
        py::arg("cell_size"),
        py::arg("dt"),
        py::arg("rho"),
        py::arg("angle_scalor"),
        py::arg("node_start") = 0,
        py::arg("node_count") = -1
    );
    m.def("value_iteration", &value_iteration_cuda, "Value iteration kernel (CUDA)");
    m.def("value_iteration_chunk", &value_iteration_chunk_cuda, "Chunked value iteration step (CUDA)");
}

