#include "msa-block-ids.cuh"

#include <climits>

#define CUDA_MSA_BLOCK_IDS_BLOCK_SIZE 256

static __global__ void k_msa_block_ids_to_rows(
        const int32_t * block_ids,
        int32_t * rows,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t nb00,
        int64_t nb03,
        int64_t nb0,
        int64_t nb1,
        int64_t nb2,
        int32_t block_size) {
    const int64_t n = ne0*ne1*ne2;

    for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x*gridDim.x) {
        const int64_t i0 = i % ne0;
        const int64_t i1 = (i / ne0) % ne1;
        const int64_t i2 = (i / (ne0*ne1)) % ne2;

        const int64_t itop = i0 / block_size;
        const int64_t ib   = i0 - itop*block_size;

        const int32_t block_id = block_ids[itop*nb00 + i2*nb03];

        rows[i0*nb0 + i1*nb1 + i2*nb2] = block_id*block_size + ib;
    }
}

void ggml_cuda_op_msa_block_ids_to_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    const int32_t n_head     = ggml_get_op_params_i32(dst, 1);

    GGML_ASSERT(block_size > 0);
    GGML_ASSERT(n_head == dst->ne[1]);
    GGML_ASSERT(src0->ne[1] == 1);
    GGML_ASSERT(src0->ne[2] == 1);
    GGML_ASSERT(src0->ne[3] == dst->ne[2]);
    GGML_ASSERT(src0->ne[0]*block_size == dst->ne[0]);

    const int64_t nb00 = src0->nb[0] / sizeof(int32_t);
    const int64_t nb03 = src0->nb[3] / sizeof(int32_t);
    const int64_t nb0  = dst->nb[0]  / sizeof(int32_t);
    const int64_t nb1  = dst->nb[1]  / sizeof(int32_t);
    const int64_t nb2  = dst->nb[2]  / sizeof(int32_t);

    const int64_t n = ggml_nelements(dst);
    const int64_t n_blocks = (n + CUDA_MSA_BLOCK_IDS_BLOCK_SIZE - 1) / CUDA_MSA_BLOCK_IDS_BLOCK_SIZE;
    const dim3 block_nums(MIN(n_blocks, (int64_t) INT_MAX), 1, 1);
    const dim3 block_dims(CUDA_MSA_BLOCK_IDS_BLOCK_SIZE, 1, 1);

    const ggml_cuda_kernel_launch_params launch_params = { block_nums, block_dims, 0, ctx.stream() };
    ggml_cuda_kernel_launch(k_msa_block_ids_to_rows, launch_params,
        (const int32_t *) src0->data, (int32_t *) dst->data,
        dst->ne[0], dst->ne[1], dst->ne[2],
        nb00, nb03, nb0, nb1, nb2, block_size);
}
