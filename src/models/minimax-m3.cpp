#include "models.h"

#include "llama-kv-cache-dsa.h"

#include <algorithm>
#include <cstdlib>

// MiniMax-M3, text-only: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling) and
// swigluoai activation. MSA is opt-in for decode; vision tower and MTP are dropped.

static bool minimax_m3_msa_enabled() {
    const char * value = getenv("LLAMA_MINIMAX_M3_MSA");
    return value && atoi(value) != 0;
}

void llama_model_minimax_m3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k, false);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_minimax_m3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_embd_indexer  = (int64_t) hparams.indexer_head_size * hparams.indexer_n_head;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        // per-head QK-norm: a single head_dim vector applied to every head
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        if (n_embd_indexer > 0) {
            layer.indexer_q_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i), {hparams.indexer_head_size}, TENSOR_NOT_REQUIRED);
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {hparams.indexer_head_size}, TENSOR_NOT_REQUIRED);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {n_embd, n_embd_indexer}, TENSOR_NOT_REQUIRED);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i), {n_embd, n_embd_indexer}, TENSOR_NOT_REQUIRED);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            // leading dense layers
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            // routed experts
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            // shared expert
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_minimax_m3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_minimax_m3::graph::build_msa_attn(
        const llama_model & model,
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * attn_inp,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        float kq_scale,
        int il) const {
    static constexpr int MSA_BLOCK_SIZE = 128;

    const auto & layer = model.layers[il];
    const int64_t n_indexer_head      = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;

    const auto * mctx_mla = inp->mctx->get_mla();
    const auto * mctx_lid = inp->mctx->get_lid();

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, k_cur, inp->get_k_idxs_mla(), il));
    ggml_build_forward_expand(gf, mctx_mla->cpy_v(ctx0, v_cur, inp->get_v_idxs_mla(), il));

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, attn_inp);
    cb(indexer_q, "indexer_q", il);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);
    indexer_q = build_norm(indexer_q, layer.indexer_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(indexer_q, "indexer_q_normed", il);

    ggml_tensor * indexer_k = build_lora_mm(layer.indexer_attn_k, attn_inp);
    cb(indexer_k, "indexer_k", il);
    indexer_k = ggml_reshape_3d(ctx0, indexer_k, n_embd_indexer_head, n_indexer_head, n_tokens);
    indexer_k = build_norm(indexer_k, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(indexer_k, "indexer_k_normed", il);

    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, inp->get_k_idxs_lid(), il));

    indexer_k = mctx_lid->get_k(ctx0, il);

    const auto n_stream = indexer_k->ne[3];
    GGML_ASSERT(n_stream == 1);
    GGML_ASSERT(indexer_q->ne[2] % n_stream == 0);

    indexer_q = ggml_view_4d(ctx0,
            indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);

    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
    cb(indexer_q, "indexer_q", il);
    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
    cb(indexer_k, "indexer_k", il);

    ggml_tensor * indexer_score = ggml_mul_mat(ctx0, indexer_k, indexer_q);
    cb(indexer_score, "indexer_kq", il);

    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
    cb(indexer_score, "indexer_score", il);

    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 0, 2, 1, 3));
    indexer_score = ggml_pool_2d(ctx0, indexer_score, GGML_OP_POOL_MAX, 1, n_indexer_head, 1, n_indexer_head, 0, 0);
    cb(indexer_score, "indexer_score_head_max", il);

    GGML_ASSERT(indexer_score->ne[0] % MSA_BLOCK_SIZE == 0);
    indexer_score = ggml_pool_2d(ctx0, indexer_score, GGML_OP_POOL_MAX, MSA_BLOCK_SIZE, 1, MSA_BLOCK_SIZE, 1, 0, 0);
    cb(indexer_score, "indexer_score_block_max", il);

    ggml_tensor * block_mask = inp->get_kq_mask_lid();
    GGML_ASSERT(block_mask->ne[0] % MSA_BLOCK_SIZE == 0);
    block_mask = ggml_pool_2d(ctx0, block_mask, GGML_OP_POOL_MAX, MSA_BLOCK_SIZE, 1, MSA_BLOCK_SIZE, 1, 0, 0);
    indexer_score = ggml_add(ctx0, indexer_score, block_mask);
    cb(indexer_score, "indexer_score_masked", il);

    const int64_t n_top_k = std::min<int64_t>(hparams.indexer_top_k, indexer_score->ne[0]);
    ggml_tensor * top_k_blocks = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, (int) n_top_k));
    cb(top_k_blocks, "top_k_blocks", il);

    ggml_tensor * rows_kv = ggml_msa_block_ids_to_rows(ctx0, top_k_blocks, MSA_BLOCK_SIZE, n_head_kv);
    cb(rows_kv, "msa_rows_kv", il);

    ggml_tensor * k = ggml_get_rows(ctx0, mctx_mla->get_k(ctx0, il), rows_kv);
    cb(k, "msa_k", il);
    ggml_tensor * v = ggml_get_rows(ctx0, mctx_mla->get_v(ctx0, il), rows_kv);
    cb(v, "msa_v", il);

    ggml_tensor * rows_mask = ggml_msa_block_ids_to_rows(ctx0, top_k_blocks, MSA_BLOCK_SIZE, 1);
    cb(rows_mask, "msa_rows_mask", il);

    ggml_tensor * kq_mask = inp->get_kq_mask_mla();
    kq_mask = ggml_view_4d(ctx0,
            kq_mask,
            1, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[3],
            kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[2], 0);

    kq_mask = ggml_get_rows(ctx0, kq_mask, rows_mask);
    kq_mask = ggml_view_4d(ctx0,
            kq_mask,
            kq_mask->ne[1], kq_mask->ne[2], 1, kq_mask->ne[3],
            kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], 0);
    if (cparams.flash_attn) {
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
    }
    cb(kq_mask, "msa_kq_mask", il);

    ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(cur, "kqv_out", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);

    if (layer.wo_b) {
        cur = ggml_add(ctx0, cur, layer.wo_b);
    }

    return cur;
}

llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // partial rotary: head_dim != n_rot, so don't assert n_embd_head == n_rot

    // swigluoai params, shared by dense and expert FFNs
    const float swiglu_alpha = 1.702f;
    const float swiglu_limit = 7.0f;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    const bool use_msa_cache = minimax_m3_msa_enabled() &&
            hparams.indexer_n_head > 0 && hparams.indexer_head_size > 0 && hparams.indexer_top_k > 0;

    llm_graph_input_attn_kv * inp_attn = nullptr;
    llm_graph_input_attn_k_dsa * inp_attn_msa = nullptr;

    if (use_msa_cache) {
        inp_attn_msa = build_attn_inp_k_dsa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // self-attention
        {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * attn_inp = cur;

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // per-head QK RMSNorm (weights already include Gemma's +1)
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // partial rotary: only the first n_rot dims are rotated
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            const bool layer_has_msa =
                model.layers[il].indexer_q_norm &&
                model.layers[il].indexer_k_norm &&
                model.layers[il].indexer_attn_q_b &&
                model.layers[il].indexer_attn_k;

            const bool use_msa_layer = inp_attn_msa && layer_has_msa &&
                n_tokens == 1 &&
                inp_attn_msa->get_kq_mask_mla()->ne[3] == 1 &&
                inp_attn_msa->get_kq_mask_mla()->ne[0] % 128 == 0 &&
                inp_attn_msa->get_kq_mask_lid()->ne[0] % 128 == 0;

            if (use_msa_layer) {
                cur = build_msa_attn(model, inp_attn_msa, attn_inp, Qcur, Kcur, Vcur, 1.0f/sqrtf(float(n_embd_head)), il);
            } else if (inp_attn_msa) {
                cur = build_attn(inp_attn_msa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // leading dense FFN (swigluoai)
            ggml_tensor * g = build_lora_mm(model.layers[il].ffn_gate, cur);
            ggml_tensor * u = build_lora_mm(model.layers[il].ffn_up,   cur);
            g   = ggml_swiglu_oai(ctx0, g, u, swiglu_alpha, swiglu_limit);
            cur = build_lora_mm(model.layers[il].ffn_down, g);
            cb(cur, "ffn_out", il);
        } else {
            // routed experts (swigluoai MoE)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (swigluoai)
            ggml_tensor * sg = build_lora_mm(model.layers[il].ffn_gate_shexp, cur);
            ggml_tensor * su = build_lora_mm(model.layers[il].ffn_up_shexp,   cur);
            sg = ggml_swiglu_oai(ctx0, sg, su, swiglu_alpha, swiglu_limit);
            ggml_tensor * ffn_shexp = build_lora_mm(model.layers[il].ffn_down_shexp, sg);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
