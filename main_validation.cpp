#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace std;
static inline vector<float> load_bin(const string &path) {
    ifstream f(path, ios::binary);
    if (!f.good()) return vector<float>{};
    f.seekg(0, ios::end);
    size_t bytes = (size_t)f.tellg();
    f.seekg(0, ios::beg);
    vector<float> data(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}


static inline auto compare_tensors(const vector<float>& ref, const vector<float>& out, const string& name, float tol= 0.1f) {
    if (ref.empty() || out.empty()) {
        cout << name << ": missing data for comparison\n";
        return;
    }
    if (ref.size() != out.size()) {
        cout << name << ": size mismatch (" << ref.size() << " vs " << out.size() << ")\n";
        return;
    }
    float difer = 0.f;
    int dif_count = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        dif_count += fabs((ref[i] - out[i])/ref[i]) > tol ? 1 : 0;
    dif_count = (100*dif_count)/ref.size();
    cout << name << " diff=" << dif_count << "% \n";
}

static inline vector<float> matmul_transpB(const vector<float>& A, const vector<float>& B, int M, int K, int N) {
    vector<float> C(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i*N + j] += A[i * K + k] * B[j * K + k];
            }
        }
    }
    return C;
}

static inline vector<float> matmul(const vector<float>& A, const vector<float>& B, int M, int K, int N) {
    vector<float> C(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    return C;
}

static inline vector<float> matmul_and_pack(const vector<float>& A, const vector<float>& B, int M, int K, int H, int Dh) {
    vector<float> C(M * H * Dh, 0.0f);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < M; ++i) {
            for (int h = 0; h < Dh; ++h) {
                for (int k = 0; k < K; ++k) {
                    C[(j*M + i) * Dh + h] += A[i * K + k] * B[(j*Dh + h) * K + k];
                }
            }
        }
    }
    return C;
}

static inline vector<float> create_random_matrix(int rows, int cols, mt19937& rng, float scale=0.02f) {
    normal_distribution<float> dist(0.0f, scale);
    vector<float> M(rows * cols);
    for (int i = 0; i < rows * cols; ++i) M[i] = dist(rng);
    return M;
}

static inline vector<int> create_random_tokens(int seq_len, int vocab_size, mt19937& rng) {
    uniform_int_distribution<int> dist(0, vocab_size - 1);
    vector<int> tokens(seq_len);
    for (int i = 0; i < seq_len; ++i) tokens[i] = dist(rng);
    return tokens;
}

static inline vector<float> embedding_lookup(const vector<int>& tokens, const vector<float>& emb_table, int vocab_size, int emb_dim) {
    int seq_len = (int)tokens.size();
    vector<float> out(seq_len * emb_dim);
    for (int i = 0; i < seq_len; ++i) {
        int t = tokens[i];
        const float* src = &emb_table[t * emb_dim];
        float* dst = &out[i * emb_dim];
        for (int d = 0; d < emb_dim; ++d) dst[d] = src[d];
    }
    return out;
}

static inline vector<float> mul(const vector<float>& A, const vector<float>& B) {
    int n = (int)A.size();
    vector<float> C(n);
    for (int i = 0; i < n; ++i) C[i] = A[i] * B[i];
    return C;
}

static inline vector<float> rmsnorm(const vector<float>& X, const vector<float>& g, int seq_len, int emb_dim, float eps=1e-6f) {
    vector<float> Y(seq_len * emb_dim);
    for (int i = 0; i < seq_len; ++i) {
        const float* row = &X[i * emb_dim];
        float sumsq = 0.0f;
        for (int d = 0; d < emb_dim; ++d) {
            float v = row[d];
            sumsq += v * v;
        }
        float rms = sqrtf(sumsq / emb_dim + eps);
        float scale = 1.0f / rms;
        for (int d = 0; d < emb_dim; ++d) Y[i * emb_dim + d] = row[d] * scale * g[d];
    }
    return Y;
}

static inline float dot_prod(const float* a, const float* b, int len) {
    float s = 0.0f;
    for (int i = 0; i < len; ++i) s += a[i] * b[i];
    return s;
}

static inline void softmax_inplace(vector<float>& scores, int offset, int len) {
    float m = -INFINITY;
    for (int i = 0; i < len; ++i) m = max(m, scores[offset + i]);
    float s = 0.0f;
    for (int i = 0; i < len; ++i) {
        float v = expf(scores[offset + i] - m);
        scores[offset + i] = v;
        s += v;
    }
    float inv = 1.0f / s;
    for (int i = 0; i < len; ++i) scores[offset + i] *= inv;
}

// duplicate kv tensors along head dimension to match q_heads
static inline vector<float> repeat_kv_heads(const vector<float>& X,
                                            int seq_len,
                                            int kv_heads,
                                            int kv_head_dim,
                                            int head_ratio) {
    int q_heads = kv_heads * head_ratio;
    vector<float> out(seq_len * q_heads * kv_head_dim);
    int block_size = kv_head_dim * seq_len;

    for (int h = 0; h < kv_heads; ++h) {
        const float* src = &X[h * block_size];
        for (int r = 0; r < head_ratio; ++r) {
            int dst_h = h * head_ratio + r;
            float* dst = &out[dst_h * block_size];
            memcpy(dst, src, block_size * sizeof(float));
        }
    }

    return out;
}


static inline vector<float> transpose(const vector<float>& X, int rows, int cols) {
    vector<float> out(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[j * rows + i] = X[i * cols + j];
    return out;
}

static inline void apply_rope(vector<float>& buf, const vector<float>& cosbuf, const vector<float>& sinbuf,
                               int heads, int head_dim, int seq_len) {
    int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        int head_base = h * seq_len * head_dim;
        for (int pos = 0; pos < seq_len; ++pos) {
            int base = head_base + pos * head_dim;
            int cosbase = pos * head_dim;
            for (int i = 0; i < half; ++i) {
                float x1 = buf[base + i];
                float x2 = buf[base + i + half];
                float cosv_low = cosbuf[cosbase + i];
                float sinv_low = sinbuf[cosbase + i];
                float cosv_high = cosbuf[cosbase + i + half];
                float sinv_high = sinbuf[cosbase + i + half];

                float y1 = x1 * cosv_low - x2 * sinv_low;
                float y2 = x2 * cosv_high + x1 * sinv_high;

                buf[base + i] = y1;
                buf[base + i + half] = y2;
            }
        }
    }
}

static inline void strided_store(const vector<float>& src, float* dst, int rows, int row_stride, int cols) {
    for (int i = 0; i < rows; ++i) {
        const float* src_row = &src[i * cols];
        float* dst_row = &dst[i * row_stride];
        memcpy(dst_row, src_row, cols * sizeof(float));
    }
}

// change to respect diferent head count for q and k/v
static inline vector<float> multi_head_attention(const vector<float>& X,
                                                 const vector<float>& Wq,
                                                 const vector<float>& Wk,
                                                 const vector<float>& Wv,
                                                 const vector<float>& Wo,
                                                 const vector<float>& sin_cache,
                                                 const vector<float>& cos_cache,
                                                 const vector<float>& mask,
                                                 int seq_len,
                                                 int emb_dim,
                                                 int q_heads, int kv_heads,
                                                 const vector<float>& q_out_ref,
                                                 const vector<float>& k_out_ref,
                                                 const vector<float>& v_out_ref,
                                                 const vector<float>& q_out_rot_ref,
                                                 const vector<float>& k_out_rot_ref,
                                                 const vector<float>& attn_output_ref,
                                                 const vector<float>& attn_output_reshaped_ref,
                                                 const vector<float>& key_repeted_ref,
                                                 const vector<float>& value_repeted_ref,
                                                 const vector<float>& attn_weights_ref,
                                                 const vector<float>& attn_weights_softmax_ref) {
    int q_head_dim = emb_dim / q_heads;
    int kv_head_dim = seq_len / kv_heads;
    int head_ratio = q_heads / kv_heads;
    const float tol = 0.01f;

    vector<float> Q = matmul_and_pack(X, Wq, seq_len, emb_dim, q_heads, q_head_dim);
    compare_tensors(q_out_ref, Q, "Q output", tol);
    vector<float> K = matmul_and_pack(X, Wk, seq_len, emb_dim, kv_heads, kv_head_dim);
    compare_tensors(k_out_ref, K, "K output", tol);
    vector<float> V = matmul_and_pack(X, Wv, seq_len, emb_dim, kv_heads, kv_head_dim);
    compare_tensors(v_out_ref, V, "V output", tol);

    // --- RoPE using cos_cache and sin_cache ---
    apply_rope(Q, cos_cache, sin_cache, q_heads, q_head_dim, seq_len);
    apply_rope(K, cos_cache, sin_cache, kv_heads, kv_head_dim, seq_len);
    compare_tensors(q_out_rot_ref, Q, "Q after RoPE", tol);
    compare_tensors(k_out_rot_ref, K, "K after RoPE", tol);

    // reshape and duplicate KV heads
    vector<float> K_rep = repeat_kv_heads(K, seq_len, kv_heads, kv_head_dim, head_ratio);
    vector<float> V_rep = repeat_kv_heads(V, seq_len, kv_heads, kv_head_dim, head_ratio);

    compare_tensors(key_repeted_ref, K_rep, "K repeated heads", tol);
    compare_tensors(value_repeted_ref, V_rep, "V repeated heads", tol);

    vector<float> out_heads(seq_len * emb_dim, 0.0f);
    int block_size = seq_len * q_head_dim;

    for (int h = 0; h < q_heads; ++h) {
        // attention scores: [seq_len, seq_len]
        vector<float> attn = matmul_transpB(
            vector<float>(&Q[h * block_size],
                          &Q[(h + 1) * block_size]),
            vector<float>(&K_rep[h * block_size],
                          &K_rep[(h + 1) * block_size]),
            seq_len, q_head_dim, seq_len);

        for (int i = 0; i < seq_len * seq_len; ++i)
            attn[i] = attn[i] / sqrtf((float)q_head_dim) + mask[i];

        for (int i = 0; i < seq_len; ++i)
            softmax_inplace(attn, i * seq_len, seq_len);

        // weighted sum: [seq_len, 64]
        vector<float> out_head = matmul(attn,
            vector<float>(&V_rep[h * block_size],
                          &V_rep[(h + 1) * block_size]),
            seq_len, seq_len, kv_head_dim);

        strided_store(out_head, &out_heads[h * q_head_dim], seq_len, emb_dim, q_head_dim);
    }
    compare_tensors(attn_output_reshaped_ref, out_heads, "Attention output reshaped", tol);
    vector<float> out = matmul_transpB(out_heads, Wo, seq_len, emb_dim, emb_dim);
    return out;
}


static inline void silu_inplace(vector<float>& M) {
    int n = (int)M.size();
    for (int i = 0; i < n; ++i) {
        float x = M[i];
        float s = 1.0f / (1.0f + expf(-x));
        M[i] = x * s;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " input_size vocab_size emb_dim N\n";
        return 1;
    }
    int arg_seq_len = stoi(argv[1]);
    int arg_vocab = stoi(argv[2]);
    int arg_emb_dim = stoi(argv[3]);
    int arg_N = stoi(argv[4]);
    int q_heads = 32;
    int kv_heads = 8;
    int mlp_dim = 8192;

    const string weights_root = "weights";
    auto file_exists = [&](const string &p) {
        ifstream f(p, ios::binary);
        return f.good();
    };

    // Infer sizes if weights exist
    int seq_len = arg_seq_len, vocab_size = arg_vocab, emb_dim = arg_emb_dim, N = arg_N;
    bool have_weights = file_exists(weights_root + "/0/inp.bin");
    if (have_weights) {
        auto q0 = load_bin(weights_root + "/0/q_proj.bin");
        emb_dim = sqrt((double)q0.size());
        auto inp0 = load_bin(weights_root + "/0/inp.bin");
        seq_len = inp0.size() / emb_dim;
        N = 0;
        while (file_exists(weights_root + "/" + to_string(N) + "/inp.bin")) ++N;
        cout << "Inferred seq_len=" << seq_len << " emb_dim=" << emb_dim << " layers=" << N << "\n";
    }

    while (q_heads > 1 && emb_dim % q_heads != 0) --q_heads;
    while (kv_heads > 1 && emb_dim % kv_heads != 0) --kv_heads;

    // Initialize or load input
    vector<float> residual;
    if (have_weights) residual = load_bin(weights_root + "/0/inp.bin");
    else {
        mt19937 rng(42);
        vector<int> tokens = create_random_tokens(seq_len, vocab_size, rng);
        vector<float> emb = create_random_matrix(vocab_size, emb_dim, rng, 0.02f);
        residual = embedding_lookup(tokens, emb, vocab_size, emb_dim);
    }

    vector<float> global_rms_g(emb_dim, 1.0f);
    const float tol = 0.01f;

    for (int layer = 0; layer < N; ++layer) {
        cout << "\n=== LAYER " << layer << " ===\n";
        string base = weights_root + "/" + to_string(layer) + "/";

        auto inp = load_bin(base + "inp.bin");
        auto attn_inp_ref = load_bin(base + "attn_inp.bin");
        auto attn_out_ref = load_bin(base + "attn_out.bin");
        auto mlp_inp_ref = load_bin(base + "mlp_inp.bin");
        auto mlp_out_ref = load_bin(base + "mlp_out.bin");
        auto out_ref = load_bin(base + "out.bin");
        auto inp_norm2 = load_bin(base + "inp_norm2.bin");
        auto q_out = load_bin(base + "q_out.bin");
        auto k_out = load_bin(base + "k_out.bin");
        auto v_out = load_bin(base + "v_out.bin");
        auto q_out_rot = load_bin(base + "q_out_rotary.bin");
        auto k_out_rot = load_bin(base + "k_out_rotary.bin");
        auto attn_output_ref = load_bin(base + "attn_weights_softmax_dropout_matmul.bin");
        auto attn_output_reshaped_ref = load_bin(base + "attn_weights_softmax_dropout_matmul_reshaped.bin");
        auto gate_proj_out = load_bin(base + "gate_proj_out.bin");
        auto up_proj_out = load_bin(base + "up_proj_out.bin");
        auto key_repeted_ref = load_bin(base + "key_states_repeated.bin");
        auto value_repeted_ref = load_bin(base + "value_states_repeated.bin");
        auto attn_weights_ref = load_bin(base + "attn_weights.bin");
        auto attn_weights_softmax_ref = load_bin(base + "attn_weights_softmax.bin");

        auto Wq = load_bin(base + "q_proj.bin");
        auto Wk = load_bin(base + "k_proj.bin");
        auto Wv = load_bin(base + "v_proj.bin");
        auto Wo = load_bin(base + "o_proj.bin");
        auto W_gate = load_bin(base + "gate_proj.bin");
        auto W_up = load_bin(base + "up_proj.bin");
        auto W_down = load_bin(base + "down_proj.bin");
        auto sin_cache = load_bin(base + "sin.bin");
        auto cos_cache = load_bin(base + "cos.bin");
        auto rms_g1 = load_bin(base + "norm1_weight.bin");
        auto rms_g2 = load_bin(base + "norm2_weight.bin");
        auto mask = load_bin(base + "mask.bin");

        // 1. RMSNorm: inp → attn_inp
        auto attn_inp = rmsnorm(inp, rms_g1, seq_len, emb_dim);
        compare_tensors(attn_inp_ref, attn_inp, "attn_inp", tol);

        // 2. Attention: attn_inp → attn_out
        auto attn_out = multi_head_attention(attn_inp_ref, Wq, Wk, Wv, Wo, sin_cache, cos_cache, mask, seq_len, emb_dim, q_heads, kv_heads,
             q_out, k_out, v_out, q_out_rot, k_out_rot, attn_output_ref, attn_output_reshaped_ref, key_repeted_ref, value_repeted_ref,
             attn_weights_ref, attn_weights_softmax_ref);
        compare_tensors(attn_out_ref, attn_out, "attn_out", tol);

        // 3. Residual add: inp + attn_out
        vector<float> after_attn(seq_len * emb_dim);
        for (int i = 0; i < seq_len * emb_dim; ++i)
            after_attn[i] = inp[i] + attn_out_ref[i];
        compare_tensors(inp_norm2, after_attn, "after_attn", tol);

        // 4. RMSNorm: after_attn → mlp_inp
        auto mlp_inp = rmsnorm(inp_norm2, rms_g2, seq_len, emb_dim);
        compare_tensors(mlp_inp_ref, mlp_inp, "mlp_inp", tol);

        // 5. MLP: mlp_inp → mlp_out
        int D_mlp = W_gate.size() / emb_dim;
        auto gate = matmul_transpB(mlp_inp_ref, W_gate, seq_len, emb_dim, D_mlp);
        auto up = matmul_transpB(mlp_inp_ref, W_up, seq_len, emb_dim, D_mlp);
        compare_tensors(up_proj_out, up, "up_proj_out", tol);
        silu_inplace(gate);
        compare_tensors(gate_proj_out, gate, "gate_proj_out", tol);
        auto fused = mul(gate, up);
        auto mlp_out = matmul_transpB(fused, W_down, seq_len, D_mlp, emb_dim);
        compare_tensors(mlp_out_ref, mlp_out, "mlp_out", tol);

        // 6. Residual add: after_attn + mlp_out → out
        vector<float> out(seq_len * emb_dim);
        for (int i = 0; i < seq_len * emb_dim; ++i)
            out[i] = after_attn[i] + mlp_out_ref[i];
        compare_tensors(out_ref, out, "out", tol);

        residual = move(out);
    }

    cout << "\nAll layers validated.\n";
    return 0;
}
