#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;

static inline vector<float> matmul(const vector<float>& A, const vector<float>& B, int A_rows, int A_cols, int B_cols) {
    vector<float> C(A_rows * B_cols);
    for (int i = 0; i < A_rows; ++i) {
        for (int k = 0; k < A_cols; ++k) {
            float a = A[i * A_cols + k];
            int b_row = k * B_cols;
            int c_row = i * B_cols;
            for (int j = 0; j < B_cols; ++j) {
                C[c_row + j] += a * B[b_row + j];
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

static inline vector<float> add(const vector<float>& A, const vector<float>& B) {
    int n = (int)A.size();
    vector<float> C(n);
    for (int i = 0; i < n; ++i) C[i] = A[i] + B[i];
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

static inline vector<float> multi_head_attention(const vector<float>& X,
                                                 const vector<float>& Wq,
                                                 const vector<float>& Wk,
                                                 const vector<float>& Wv,
                                                 const vector<float>& Wo,
                                                 int seq_len,
                                                 int emb_dim,
                                                 int num_heads) {
    int head_dim = emb_dim / num_heads;
    vector<float> Q = matmul(X, Wq, seq_len, emb_dim, emb_dim);
    vector<float> K = matmul(X, Wk, seq_len, emb_dim, emb_dim);
    vector<float> V = matmul(X, Wv, seq_len, emb_dim, emb_dim);

    // --- RoPE (rotary positional embedding) applied to Q and K in-place ---
    int half = head_dim / 2; // number of complex pairs per head
    vector<float> inv_freq(half);
    for (int k = 0; k < half; ++k) {
        inv_freq[k] = 1.0f / powf(10000.0f, (2.0f * k) / (float)head_dim);
    }

    for (int h = 0; h < num_heads; ++h) {
        int head_offset = h * head_dim;
        for (int pos = 0; pos < seq_len; ++pos) {
            int base = pos * emb_dim + head_offset;
            for (int k = 0; k < half; ++k) {
                int i0 = base + 2 * k;
                int i1 = i0 + 1;
                float angle = pos * inv_freq[k];
                float cs = cosf(angle);
                float sn = sinf(angle);

                // rotate Q pair
                float q0 = Q[i0];
                float q1 = Q[i1];
                Q[i0] = q0 * cs - q1 * sn;
                Q[i1] = q0 * sn + q1 * cs;

                // rotate K pair
                float k0 = K[i0];
                float k1 = K[i1];
                K[i0] = k0 * cs - k1 * sn;
                K[i1] = k0 * sn + k1 * cs;
            }
            // if head_dim is odd, leave the last channel unchanged
        }
    }
    // --- end RoPE ---

    vector<float> out_heads(seq_len * emb_dim);
    vector<float> scores(seq_len);
    for (int h = 0; h < num_heads; ++h) {
        int head_offset = h * head_dim;
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                const float* q = &Q[i * emb_dim + head_offset];
                const float* k = &K[j * emb_dim + head_offset];
                float sc = dot_prod(q, k, head_dim);
                sc /= sqrtf((float)head_dim);
                // causal mask (LLaMA style): prevent attending to future tokens
                if (j > i) sc -= 1e9f;
                scores[j] = sc;
            }
            softmax_inplace(scores, 0, seq_len);
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; ++j)
                    acc += scores[j] * V[j * emb_dim + head_offset + d];
                out_heads[i * emb_dim + head_offset + d] = acc;
            }
        }
    }

    vector<float> out = matmul(out_heads, Wo, seq_len, emb_dim, emb_dim);
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
    int seq_len = stoi(argv[1]);
    int vocab_size = stoi(argv[2]);
    int emb_dim = stoi(argv[3]);
    int N = stoi(argv[4]);
    int num_heads = 8;
    while (num_heads > 1 && emb_dim % num_heads != 0) --num_heads;
    int head_dim = emb_dim / num_heads;
    mt19937 rng(42);
    vector<int> tokens = create_random_tokens(seq_len, vocab_size, rng);
    vector<float> emb_table = create_random_matrix(vocab_size, emb_dim, rng, 0.02f);
    vector<float> X = embedding_lookup(tokens, emb_table, vocab_size, emb_dim);
    vector<float> pos = create_random_matrix(seq_len, emb_dim, rng, 0.02f);
    for (int i = 0; i < seq_len * emb_dim; ++i) X[i] += pos[i];
    vector<float> rms_g(emb_dim, 1.0f);
    vector<float> Wq = create_random_matrix(emb_dim, emb_dim, rng);
    vector<float> Wk = create_random_matrix(emb_dim, emb_dim, rng);
    vector<float> Wv = create_random_matrix(emb_dim, emb_dim, rng);
    vector<float> Wo = create_random_matrix(emb_dim, emb_dim, rng);
    int hidden = max(emb_dim * 4, 1);
    vector<float> W_gate = create_random_matrix(emb_dim, hidden, rng, 0.02f);
    vector<float> W_up = create_random_matrix(emb_dim, hidden, rng, 0.02f);
    vector<float> W_down = create_random_matrix(hidden, emb_dim, rng, 0.02f);
    vector<float> residual = X;
    for (int iter = 0; iter < N; ++iter) {
        vector<float> x1 = rmsnorm(residual, rms_g, seq_len, emb_dim);
        vector<float> attn_out = multi_head_attention(x1, Wq, Wk, Wv, Wo, seq_len, emb_dim, num_heads);
        vector<float> added = add(attn_out, residual);
        vector<float> x2 = rmsnorm(added, rms_g, seq_len, emb_dim);
        vector<float> gate = matmul(x2, W_gate, seq_len, emb_dim, hidden);
        vector<float> up = matmul(x2, W_up, seq_len, emb_dim, hidden);
        silu_inplace(gate);
        vector<float> fused = add(gate, up);
        vector<float> mlp_out = matmul(fused, W_down, seq_len, hidden, emb_dim);
        residual = add(mlp_out, added);
    }
    cout.setf(ios::fixed);
    cout.precision(6);
    for (int i = 0; i < min(5, seq_len); ++i) {
        cout << "token " << i << ": ";
        for (int d = 0; d < min(8, emb_dim); ++d) cout << residual[i * emb_dim + d] << " ";
        cout << "\n";
    }
    return 0;
}
