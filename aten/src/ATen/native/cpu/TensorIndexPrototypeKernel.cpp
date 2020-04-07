
#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {
namespace {

void assert_special_case(const Tensor& self, const Tensor& index, int64_t dim){
    TORCH_CHECK_INDEX(index.dim() == 1, "index must be 1D.");
    TORCH_CHECK_INDEX(self.dim() == 2, "self must be 2D.");
    TORCH_CHECK(dim == 0 || dim == 1, "dim must be zero or one.")
    TORCH_CHECK(self.is_contiguous(), "implementation assumes self is contiguous.")
}

void index_select_memcpy_dim0_numel_outer_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/0);
    constexpr int64_t dim = 0;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    auto self_dim_size = self.size(dim);

    const size_t type_size = elementSize(self.scalar_type());
    const size_t feature_len = sizes[1 - dim];
    const size_t copy_bytes = feature_len * type_size;

    auto index_buffer = index.data_ptr<int64_t>();
    auto self_buffer = static_cast<char*>(self.data_ptr());
    auto result_buffer = static_cast<char*>(result.data_ptr());
    
    for (size_t i = 0; i < numel; ++i){
        auto source_i = index_buffer[i];
        TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

        auto self_data = self_buffer + source_i * copy_bytes;
        auto result_data = result_buffer + i * copy_bytes;
        memcpy(result_data, self_data, copy_bytes);
    }
}

template <bool NUMEL_OUTER>
void index_select_ptr_assign_dim0_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/0);
    constexpr int64_t dim = 0;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    auto self_dim_size = self.size(dim);
    const size_t feature_len = sizes[1 - dim];

    AT_DISPATCH_ALL_TYPES_AND(
        ScalarType::Bool, self.scalar_type(), "index_select_memcpy_dim1_kernel", [&] {
            auto index_buffer = index.data_ptr<int64_t>();
            auto self_buffer = self.data_ptr<scalar_t>();
            auto result_buffer = result.data_ptr<scalar_t>();

            if (NUMEL_OUTER) {
                for (size_t i = 0; i < numel; ++i){
                    auto source_i = index_buffer[i];
                    TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                    auto self_data = self_buffer + source_i * feature_len;
                    auto result_data = result_buffer + i * feature_len;
                    for (size_t j = 0; j < feature_len; ++j){
                        *(result_data + j) = *(self_data + j);
                    }
                }
            } else {
                for (size_t j = 0; j < feature_len; ++j){
                    for (size_t i = 0; i < numel; ++i){
                        auto source_i = index_buffer[i];
                        TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                        auto self_data = self_buffer + source_i * feature_len;
                        auto result_data = result_buffer + i * feature_len;
                        *(result_data + j) = *(self_data + j);
                    }
                }
            }
        });
}

template <bool NUMEL_OUTER>
void index_select_ptr_assign_dim1_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/1);
    constexpr int64_t dim = 1;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    const size_t self_dim_size = self.size(dim);
    const size_t feature_len = sizes[1 - dim];

    AT_DISPATCH_ALL_TYPES_AND(
        ScalarType::Bool, self.scalar_type(), "index_select_memcpy_dim1_kernel", [&] {
            auto index_buffer = index.data_ptr<int64_t>();
            auto self_buffer = self.data_ptr<scalar_t>();
            auto result_buffer = result.data_ptr<scalar_t>();

            if (NUMEL_OUTER) {
                for (int64_t i = 0; i < numel; ++i){
                    auto source_i = index_buffer[i];
                    TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                    for (int64_t j = 0; j < feature_len; ++j){
                        auto self_data = self_buffer + (source_i + j * self_dim_size);
                        auto result_data = result_buffer + (i + j * numel);
                        *result_data = *self_data;
                    }
                }
            } else {
                for (int64_t j = 0; j < feature_len; ++j){
                    for (int64_t i = 0; i < numel; ++i){
                        auto source_i = index_buffer[i];
                        TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                        auto self_data = self_buffer + (source_i + j * self_dim_size);
                        auto result_data = result_buffer + (i + j * numel);
                        *result_data = *self_data;
                    }
                }
            }
        });
}

template <bool NUMEL_OUTER>
void index_select_memcpy_dim1_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/1);
    constexpr int64_t dim = 1;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    const size_t self_dim_size = self.size(dim);
    const size_t feature_len = sizes[1 - dim];
    const size_t type_size = elementSize(self.scalar_type());

    auto index_buffer = index.data_ptr<int64_t>();
    auto self_buffer = static_cast<char*>(self.data_ptr());
    auto result_buffer = static_cast<char*>(result.data_ptr());

    if (NUMEL_OUTER) {
        for (int64_t i = 0; i < numel; ++i){
            auto source_i = index_buffer[i];
            TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

            for (int64_t j = 0; j < feature_len; ++j){
                auto self_data = self_buffer + (source_i + j * self_dim_size) * type_size;
                auto result_data = result_buffer + (i + j * numel) * type_size;
                memcpy(result_data, self_data, type_size);
            }
        }
    } else {
        for (int64_t j = 0; j < feature_len; ++j){
            for (int64_t i = 0; i < numel; ++i){
                auto source_i = index_buffer[i];
                TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                auto self_data = self_buffer + (source_i + j * self_dim_size) * type_size;
                auto result_data = result_buffer + (i + j * numel) * type_size;
                memcpy(result_data, self_data, type_size);
            }
        }
    }
}

} // namespace

REGISTER_DISPATCH(index_select_memcpy_dim0_numel_outer_kernel_stub, 
                  &index_select_memcpy_dim0_numel_outer_kernel);
REGISTER_DISPATCH(index_select_ptr_assign_dim0_kernel_true_stub,
                  &index_select_ptr_assign_dim0_kernel<true>);
REGISTER_DISPATCH(index_select_ptr_assign_dim0_kernel_false_stub,
                  &index_select_ptr_assign_dim0_kernel<false>);

REGISTER_DISPATCH(index_select_memcpy_dim1_kernel_true_stub,
                  &index_select_memcpy_dim1_kernel<true>);
REGISTER_DISPATCH(index_select_memcpy_dim1_kernel_false_stub,
                  &index_select_memcpy_dim1_kernel<false>);
REGISTER_DISPATCH(index_select_ptr_assign_dim1_kernel_true_stub,
                  &index_select_ptr_assign_dim1_kernel<true>);
REGISTER_DISPATCH(index_select_ptr_assign_dim1_kernel_false_stub,
                  &index_select_ptr_assign_dim1_kernel<false>);

}} // namespace at::native
