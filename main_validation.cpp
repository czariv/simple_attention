#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <time.h>
#include "interface.h"

#include <immintrin.h>

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


static inline auto compare_tensors(const vector<float>& ref, const vector<float>& out, const string& name, const int size, const float tol= 0.1f) {
    if (ref.empty() || out.empty()) {
        cout << name << ": missing data for comparison\n";
        return;
    }
    if (out.size() != size) {
        cout << name << ": size mismatch (" << ref.size() << " vs " << size << ")\n";
        return;
    }
    double difer = 0.f;
    int dif_count = 0;
    for (size_t i = 0; i < size; ++i){
        difer = fabs((ref[i] - out[i])/ref[i]);
        if (difer > tol)
            dif_count += 1;
    }
    dif_count = (100*dif_count)/size;
    cout << name << " diff=" << dif_count << "% \n";
}

static inline auto reshape_tensor(const vector<float>& ref, const int old_dim, const int new_dim){
    vector<float> out(new_dim * new_dim);
    for(int i = 0; i < new_dim; ++i)
        for(int j = 0; j < new_dim; ++j)
            out[i*new_dim + j] = ref[i*old_dim + j];
    return out;
}

static inline auto copy_changed_layout(const vector<float> &ref, const int N, const int K, const int stride=-1){
    vector<float> out(N*K);
    vector<float> idd(K*K);
    for(int i = 0; i < K; ++i)
        idd[i + i * K] = 1;
    gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasPre,
            N, K, K,
            1.0f,
            ref.data(), K,
            idd.data(), K,
            0.0f,
            out.data(), (stride == -1) ? K : stride,
            stride, 0);
    return out;

}

static inline void matmul(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                                    int M, int N, int K, int alpha, const float *A, int lda, const float *B, int ldb, int beta, float *C, int ldc) {
    if (alpha == 1 && beta == 0){
        gemm(Order, TransA, TransB,
                    M, N, K,
                    1.0f,
                    A, lda,
                    B, ldb,
                    0.0f,
                    C, ldc);
    } else {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                for (int j = 0; j < N; ++j) {
                    C[i*N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
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

static inline vector<float> rmsnorm(const vector<float>& X, const vector<float>& g, int seq_len, int emb_dim, float eps=1e-5f) {
    vector<float> Y(seq_len * emb_dim);

#if defined(__AVX512F__)
    const int VEC = 16;
    const __m512 eps_v = _mm512_set1_ps(eps);
    const __m512 emb_dim_v = _mm512_set1_ps((float)emb_dim);

    for (int i = 0; i < seq_len; ++i) {

        const float* row = &X[i * emb_dim];
        float* out_row   = &Y[i * emb_dim];

        __m512 acc = _mm512_setzero_ps();
        int d = 0;

        for (; d + VEC <= emb_dim; d += VEC) {
            __m512 v = _mm512_loadu_ps(row + d);
            acc = _mm512_fmadd_ps(v, v, acc);
        }

        float sumsq = _mm512_reduce_add_ps(acc);
        for (; d < emb_dim; ++d) sumsq += row[d] * row[d];

        float rms    = sqrtf(sumsq / emb_dim + eps);
        float scale  = 1.0f / rms;
        __m512 scale_v = _mm512_set1_ps(scale);

        d = 0;
        for (; d + VEC <= emb_dim; d += VEC) {
            __m512 v  = _mm512_loadu_ps(row + d);
            __m512 gv = _mm512_loadu_ps(&g[d]);
            __m512 y  = _mm512_mul_ps(_mm512_mul_ps(v, scale_v), gv);
            _mm512_storeu_ps(out_row + d, y);
        }

        for (; d < emb_dim; ++d) {
            out_row[d] = row[d] * scale * g[d];
        }
    }

#else
    for (int i = 0; i < seq_len; ++i) {

        const float* row = &X[i * emb_dim];
        float* out_row   = &Y[i * emb_dim];

        float sumsq = 0.0f;
        for (int d = 0; d < emb_dim; ++d) sumsq += row[d] * row[d];

        float rms = sqrtf(sumsq / emb_dim + eps);
        float scale = 1.0f / rms;

        for (int d = 0; d < emb_dim; ++d)
            out_row[d] = row[d] * scale * g[d];
    }
#endif

    return Y;
}

static inline vector<float> rmsnorm_packed(const vector<float>& X, const vector<float>& g, int seq_len, int emb_dim, int block_size = 448, float eps=1e-5f) {
    vector<float> Y(seq_len * emb_dim);
#if defined(__AVX512F__)
    const int B = 704;
    const int block_size_B = 352;
    const int block_size_A = block_size;
    for (int i = 0; i < seq_len; i += 4)
    {
        const float* row  = &X[i * block_size_A];
        float* out_row    = &Y[i * block_size_A];

        __m512 vsumsq = _mm512_setzero_ps();

        auto broadcast4 = [] (float x0, float x1, float x2, float x3) {
            return _mm512_set_ps(
                x3,x3,x3,x3, x2,x2,x2,x2,
                x1,x1,x1,x1, x0,x0,x0,x0
            );
        };

        // --------------------------
        // REGION A ACCUMULATION
        // --------------------------
        int offset = 0;
        int d = 0;

        for (; d < emb_dim - B; d += 4)
        {
            if (d && (d % block_size_A) == 0)
                offset += block_size_A * seq_len;

            const float* base = row + offset + (d % block_size_A) * 4;

            __m512 vx = _mm512_loadu_ps(base);

            vsumsq = _mm512_fmadd_ps(vx, vx, vsumsq);
        }

        // --------------------------
        // REGION B ACCUMULATION
        // --------------------------

        offset += (block_size_A * seq_len) - 96 * i;
        int block_size = block_size_B;

        for (int dd = 0; dd < B; dd += 4, d += 4)
        {
            if (dd && (dd % block_size) == 0)
                offset += block_size * seq_len;

            const float* base = row + offset + (dd % block_size) * 4;

            __m512 vx = _mm512_loadu_ps(base);

            vsumsq = _mm512_fmadd_ps(vx, vx, vsumsq);
        }

        // --------------------------
        // SINGLE UNPACK
        // --------------------------

        __m256 lo256 = _mm512_castps512_ps256(vsumsq);
        __m256 hi256 = _mm512_extractf32x8_ps(vsumsq, 1);
        __m256 sum256 = _mm256_add_ps(lo256, hi256);

        __m128 lo128 = _mm256_castps256_ps128(sum256);
        __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
        __m128 sum128 = _mm_add_ps(lo128, hi128);

        float inv1 = 1.0f / sqrtf(sum128[0] / emb_dim + eps);
        float inv2 = 1.0f / sqrtf(sum128[1]/ emb_dim + eps);
        float inv3 = 1.0f / sqrtf(sum128[2] / emb_dim + eps);
        float inv4 = 1.0f / sqrtf(sum128[3] / emb_dim + eps);

        __m512 v_inv = _mm512_set4_ps(inv4, inv3, inv2, inv1);
        // --------------------------
        // WRITEBACK REGION A
        // --------------------------
        offset = 0;
        block_size = block_size_A;

        for (int d = 0; d < emb_dim - B; d += 4)
        {
            if (d && (d % block_size) == 0)
                offset += block_size * seq_len;

            const float* base = row + offset + (d % block_size) * 4;
            float* out  = out_row + offset + (d % block_size) * 4;

            __m512 vx = _mm512_loadu_ps(base);

            __m512 vg = broadcast4(g[d], g[d+1], g[d+2], g[d+3]);

            __m512 scaled = _mm512_mul_ps(_mm512_mul_ps(vx, v_inv), vg);

            _mm512_storeu_ps(out, scaled);
        }

        // --------------------------
        // WRITEBACK REGION B
        // --------------------------
        offset += (block_size_A * seq_len) - 96 * i;
        block_size = block_size_B;

        for (int dd = 0; dd < B; dd += 4)
        {
            int gd = emb_dim - B + dd;

            if (dd && (dd % block_size) == 0)
                offset += block_size * seq_len;

            const float* base = row + offset + (dd % block_size) * 4;
                  float* out  = out_row + offset + (dd % block_size) * 4;

            __m512 vx = _mm512_loadu_ps(base);

            __m512 vg = broadcast4(g[gd], g[gd+1], g[gd+2], g[gd+3]);

            __m512 scaled = _mm512_mul_ps(_mm512_mul_ps(vx, v_inv), vg);

            _mm512_storeu_ps(out, scaled);
        }
    }
#else
    for (int i = 0; i < seq_len; i+=4) {
        int offset = 0;
        const float* row = &X[i * block_size];
        float* out_row   = &Y[i * block_size];
        float sumsq1 = 0.0f;
        float sumsq2 = 0.0f;
        float sumsq3 = 0.0f;
        float sumsq4 = 0.0f;
        for (int d = 0; d < (emb_dim - 704); ++d) {
            if (d%block_size == 0 && d != 0)
                offset += block_size*seq_len;
            float v1 = row[offset + (d%block_size)*4];
            float v2 = row[offset + (d%block_size)*4 + 1];
            float v3 = row[offset + (d%block_size)*4 + 2];
            float v4 = row[offset + (d%block_size)*4 + 3];
            sumsq1 += v1 * v1;
            sumsq2 += v2 * v2;
            sumsq3 += v3 * v3;
            sumsq4 += v4 * v4;
        }
        offset += block_size*seq_len - 96*(i);
        block_size = 352;
        for (int d = 0; d < 704; ++d) {
            if (d%block_size == 0 && d != 0)
                offset += block_size*seq_len;
            float v1 = row[offset + (d%block_size)*4];
            float v2 = row[offset + (d%block_size)*4 + 1];
            float v3 = row[offset + (d%block_size)*4 + 2];
            float v4 = row[offset + (d%block_size)*4 + 3];
            sumsq1 += v1 * v1;
            sumsq2 += v2 * v2;
            sumsq3 += v3 * v3;
            sumsq4 += v4 * v4;
        }
        float rms1 = sqrtf(sumsq1 / emb_dim + eps);
        float rms2 = sqrtf(sumsq2 / emb_dim + eps);
        float rms3 = sqrtf(sumsq3 / emb_dim + eps);
        float rms4 = sqrtf(sumsq4 / emb_dim + eps);
        float scale1 = 1.0f / rms1;
        float scale2 = 1.0f / rms2;
        float scale3 = 1.0f / rms3;
        float scale4 = 1.0f / rms4;
        offset = 0;
        block_size = 448;
        for (int d = 0; d < (emb_dim - 704); ++d) {
            if (d%block_size == 0 && d != 0)
                offset += block_size*seq_len;
            out_row[offset + (d%block_size)*4]     = row[offset + (d%block_size)*4]     * scale1 * g[d];
            out_row[offset + (d%block_size)*4 + 1] = row[offset + (d%block_size)*4 + 1] * scale2 * g[d];
            out_row[offset + (d%block_size)*4 + 2] = row[offset + (d%block_size)*4 + 2] * scale3 * g[d];
            out_row[offset + (d%block_size)*4 + 3] = row[offset + (d%block_size)*4 + 3] * scale4 * g[d];
        }
        offset += block_size*seq_len - 96*(i);
        block_size = 352;
        for (int d = 0; d < 704; ++d) {
            if (d%block_size == 0 && d != 0)
                offset += block_size*seq_len;
            out_row[offset + (d%block_size)*4]     = row[offset + (d%block_size)*4]     * scale1 * g[emb_dim-704 + d];
            out_row[offset + (d%block_size)*4 + 1] = row[offset + (d%block_size)*4 + 1] * scale2 * g[emb_dim-704 + d];
            out_row[offset + (d%block_size)*4 + 2] = row[offset + (d%block_size)*4 + 2] * scale3 * g[emb_dim-704 + d];
            out_row[offset + (d%block_size)*4 + 3] = row[offset + (d%block_size)*4 + 3] * scale4 * g[emb_dim-704 + d];
        }
        block_size = 448;
    }
#endif
    return Y;
}

static inline float dot_prod(const float* a, const float* b, int len) {
    float s = 0.0f;
    for (int i = 0; i < len; ++i) s += a[i] * b[i];
    return s;
}

static inline void softmax_inplace(vector<float>& scores, int offset, int len) {
    #ifdef __AVX512F__
    const float neg_inf = -std::numeric_limits<float>::infinity();
    __m512 vmax = _mm512_set1_ps(neg_inf);
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        const float* ptr = scores.data() + offset + i;
        __m512 v = _mm512_loadu_ps(ptr);
        vmax = _mm512_max_ps(vmax, v);
    }
    int rem = len - i;
    if (rem > 0) {
        __mmask16 k = (1u << rem) - 1;
        const float* ptr = scores.data() + offset + i;
        __m512 v = _mm512_maskz_loadu_ps(k, ptr);
        vmax = _mm512_max_ps(vmax, v);
    }
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, vmax);
    float m = tmp[0];
    for (int j = 1; j < 16; ++j) if (tmp[j] > m) m = tmp[j];

    float s = 0.0f;
    for (int idx = 0; idx < len; ++idx) {
        float v = expf(scores[offset + idx] - m);
        scores[offset + idx] = v;
        s += v;
    }

    float inv = 1.0f / s;
    __m512 vinv = _mm512_set1_ps(inv);
    i = 0;
    for (; i + 16 <= len; i += 16) {
        float* ptr = scores.data() + offset + i;
        __m512 v = _mm512_loadu_ps(ptr);
        v = _mm512_mul_ps(v, vinv);
        _mm512_storeu_ps(ptr, v);
    }
    rem = len - i;
    if (rem > 0) {
        __mmask16 k = (1u << rem) - 1;
        float* ptr = scores.data() + offset + i;
        __m512 v = _mm512_maskz_loadu_ps(k, ptr);
        v = _mm512_mul_ps(v, vinv);
        _mm512_mask_storeu_ps(ptr, k, v);
    }
#else
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
#endif
}

static inline void softmax_inplace_packed(vector<float>& scores, int offset, int len, int head_dim, int stride) {
#ifdef __AVX512F__
    const int block_cols = 16;
    const int subblocks = (head_dim + block_cols - 1) / block_cols;
    float m1 = -std::numeric_limits<float>::infinity();
    float m2 = -std::numeric_limits<float>::infinity();
    float m3 = -std::numeric_limits<float>::infinity();
    float m4 = -std::numeric_limits<float>::infinity();

    int base_off = offset - len * stride + stride * 3;
    int cur_off = base_off;
    int processed = 0;
    const __m512i idx_r0 = _mm512_setr_epi32(
        0,4,8,12, 16,20,24,28,
        1,5,9,13, 17,21,25,29
    );
    const __m512i idx_r1 = _mm512_setr_epi32(
        2,6,10,14, 18,22,26,30,
        3,7,11,15, 19,23,27,31
    );
    while (processed < len) {
        // start of a head_dim segment
        cur_off += (processed % stride) ? head_dim * 4 : len * stride - stride * 3;
        // process this head_dim segment (assume head_dim is multiple of 16 or handle masked loads)
        int cols = head_dim;
        int c = 0;
        for (int sb = 0; sb < subblocks; ++sb) {
            int remaining_cols = cols - c;
            int this_cols = remaining_cols >= block_cols ? block_cols : remaining_cols;
            const float* ptr = scores.data() + cur_off + c * 4;
            if (this_cols == block_cols) {
                __m512 a0 = _mm512_loadu_ps(ptr + 0);
                __m512 a1 = _mm512_loadu_ps(ptr + 16);
                __m512 a2 = _mm512_loadu_ps(ptr + 32);
                __m512 a3 = _mm512_loadu_ps(ptr + 48);

                __m512 t0 = _mm512_unpacklo_ps(a0, a2);
                __m512 t1 = _mm512_unpackhi_ps(a0, a2);
                __m512 t2 = _mm512_unpacklo_ps(a1, a3);
                __m512 t3 = _mm512_unpackhi_ps(a1, a3);

                // r0 = first 16 columns of row 0
                __m512 r0 = _mm512_permutex2var_ps(t0, idx_r0, t2);
                __m512 r1 = _mm512_permutex2var_ps(t0, idx_r1, t2);
                __m512 r2 = _mm512_permutex2var_ps(t1, idx_r0, t3);
                __m512 r3 = _mm512_permutex2var_ps(t1, idx_r1, t3);

                m1 = std::max(m1, _mm512_reduce_max_ps(r0));
                m2 = std::max(m2, _mm512_reduce_max_ps(r1));
                m3 = std::max(m3, _mm512_reduce_max_ps(r2));
                m4 = std::max(m4, _mm512_reduce_max_ps(r3));
            } else {
                // masked path for final partial subblock
                unsigned int k = (1u << (this_cols * 4)) - 1; // this_cols*4 floats in this partial chunk
                // but masks for _mm512_maskz_loadu_ps use 16-lane mask; we'll load piecewise by 16-float lanes
                // easier safe fallback: scalar scan on this partial tail
                for (int cc = 0; cc < this_cols; ++cc) {
                    float v0 = scores[cur_off + (c + cc) * 4 + 0];
                    float v1 = scores[cur_off + (c + cc) * 4 + 1];
                    float v2 = scores[cur_off + (c + cc) * 4 + 2];
                    float v3 = scores[cur_off + (c + cc) * 4 + 3];
                    if (v0 > m1) m1 = v0;
                    if (v1 > m2) m2 = v1;
                    if (v2 > m3) m3 = v2;
                    if (v3 > m4) m4 = v3;
                }
            }
            c += this_cols;
        }
        processed += head_dim;
    }

    // now scalar exp + accumulation (we keep expf scalar for correctness)
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    cur_off = base_off;
    processed = 0;
    while (processed < len) {
        cur_off += (processed % stride) ? head_dim * 4 : len * stride - stride * 3;
        int cols = head_dim;
        for (int c = 0; c < cols; ++c) {
            int idx = cur_off + c * 4;
            float v1 = expf(scores[idx + 0] - m1);
            float v2 = expf(scores[idx + 1] - m2);
            float v3 = expf(scores[idx + 2] - m3);
            float v4 = expf(scores[idx + 3] - m4);
            scores[idx + 0] = v1;
            scores[idx + 1] = v2;
            scores[idx + 2] = v3;
            scores[idx + 3] = v4;
            s1 += v1;
            s2 += v2;
            s3 += v3;
            s4 += v4;
        }
        processed += head_dim;
    }

    float inv1 = 1.0f / s1;
    float inv2 = 1.0f / s2;
    float inv3 = 1.0f / s3;
    float inv4 = 1.0f / s4;

    __m512 scale = _mm512_set4_ps(inv4, inv3, inv2, inv1);

    cur_off = base_off;
    processed = 0;

    while (processed < len) {
        // ✔ use the SAME offset logic as max and exp loops
        cur_off += (processed % stride) ? head_dim * 4 : len * stride - stride * 3;

        int c = 0;
        for (; c + 16 <= head_dim; c += 16) {
            float* ptr = scores.data() + cur_off + c*4;

            __m512 v0 = _mm512_loadu_ps(ptr +  0);
            __m512 v1 = _mm512_loadu_ps(ptr + 16);
            __m512 v2 = _mm512_loadu_ps(ptr + 32);
            __m512 v3 = _mm512_loadu_ps(ptr + 48);

            v0 = _mm512_mul_ps(v0, scale);
            v1 = _mm512_mul_ps(v1, scale);
            v2 = _mm512_mul_ps(v2, scale);
            v3 = _mm512_mul_ps(v3, scale);

            _mm512_storeu_ps(ptr +  0, v0);
            _mm512_storeu_ps(ptr + 16, v1);
            _mm512_storeu_ps(ptr + 32, v2);
            _mm512_storeu_ps(ptr + 48, v3);
        }

        // scalar tail
        for (; c < head_dim; c++) {
            float* ptr = scores.data() + cur_off + c*4;
            ptr[0] *= inv1;
            ptr[1] *= inv2;
            ptr[2] *= inv3;
            ptr[3] *= inv4;
        }

        processed += head_dim;
    }

#else
    // Original fallback
    float m1 = -INFINITY, m2 = -INFINITY, m3 = -INFINITY, m4 = -INFINITY;
    offset -= len * stride;
    int old_offset = offset;
    for (int i = 0; i < len; ++i) {
        if (i % stride == 0)
            offset += len * stride;
        m1 = std::max(m1, scores[offset + (i % stride) * 4]);
        m2 = std::max(m2, scores[offset + (i % stride) * 4 + 1]);
        m3 = std::max(m3, scores[offset + (i % stride) * 4 + 2]);
        m4 = std::max(m4, scores[offset + (i % stride) * 4 + 3]);
    }
    float s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    offset = old_offset;
    for (int i = 0; i < len; ++i) {
        if (i % stride == 0)
            offset += len * stride;
        float v1 = expf(scores[offset + (i % stride) * 4] - m1);
        float v2 = expf(scores[offset + (i % stride) * 4 + 1] - m2);
        float v3 = expf(scores[offset + (i % stride) * 4 + 2] - m3);
        float v4 = expf(scores[offset + (i % stride) * 4 + 3] - m4);
        scores[offset + (i % stride) * 4]     = v1;
        scores[offset + (i % stride) * 4 + 1] = v2;
        scores[offset + (i % stride) * 4 + 2] = v3;
        scores[offset + (i % stride) * 4 + 3] = v4;
        s1 += v1; s2 += v2; s3 += v3; s4 += v4;
    }
    float inv1 = 1.f/s1, inv2 = 1.f/s2, inv3 = 1.f/s3, inv4 = 1.f/s4;
    offset = old_offset;
    for (int i = 0; i < len; ++i) {
        if (i % stride == 0)
            offset += len * stride;
        scores[offset + (i % stride) * 4]     *= inv1;
        scores[offset + (i % stride) * 4 + 1] *= inv2;
        scores[offset + (i % stride) * 4 + 2] *= inv3;
        scores[offset + (i % stride) * 4 + 3] *= inv4;
    }
#endif
}

// duplicate kv tensors along head dimension to match q_heads
static inline vector<float> repeat_kv_heads(const vector<float>& X,
                                            int seq_len,
                                            int kv_heads,
                                            int kv_head_dim,
                                            int head_ratio) {
    int q_heads = kv_heads * head_ratio;
    vector<float> out(seq_len * q_heads * kv_head_dim);

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int h = 0; h < kv_heads; ++h) {
            int src_base = pos * kv_heads * kv_head_dim + h * kv_head_dim;
            for (int r = 0; r < head_ratio; ++r) {
                int dst_h = h * head_ratio + r;
                int dst_base = pos * q_heads * kv_head_dim + dst_h * kv_head_dim;
                memcpy(&out[dst_base], &X[src_base], kv_head_dim * sizeof(float));
            }
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
#ifdef __AVX512F__
    int half = head_dim / 2;
    for (int pos = 0; pos < seq_len; ++pos) {
        int pos_base = pos * head_dim * heads;
        int cosbase  = pos * head_dim;
        for (int h = 0; h < heads; ++h) {
            int base = pos_base + h * head_dim;
            int i = 0;
            for (; i + 16 <= half; i += 16) {
                const float* px1 = &buf[base + i];
                const float* px2 = &buf[base + i + half];
                const float* pcosl = &cosbuf[cosbase + i];
                const float* psinl = &sinbuf[cosbase + i];
                const float* pcosh = &cosbuf[cosbase + i + half];
                const float* psinh = &sinbuf[cosbase + i + half];

                __m512 x1 = _mm512_loadu_ps(px1);
                __m512 x2 = _mm512_loadu_ps(px2);
                __m512 cos_low  = _mm512_loadu_ps(pcosl);
                __m512 sin_low  = _mm512_loadu_ps(psinl);
                __m512 cos_high = _mm512_loadu_ps(pcosh);
                __m512 sin_high = _mm512_loadu_ps(psinh);

                __m512 y1 = _mm512_fmsub_ps(x1, cos_low, _mm512_mul_ps(x2, sin_low));
                __m512 y2 = _mm512_fmadd_ps(x2, cos_high, _mm512_mul_ps(x1, sin_high));

                _mm512_storeu_ps(&buf[base + i],        y1);
                _mm512_storeu_ps(&buf[base + i + half], y2);
            }
            for (; i < half; ++i) {
                float x1 = buf[base + i];
                float x2 = buf[base + i + half];
                float cos_low  = cosbuf[cosbase + i];
                float sin_low  = sinbuf[cosbase + i];
                float cos_high = cosbuf[cosbase + i + half];
                float sin_high = sinbuf[cosbase + i + half];
                float y1 = x1 * cos_low - x2 * sin_low;
                float y2 = x2 * cos_high + x1 * sin_high;
                buf[base + i]        = y1;
                buf[base + i + half] = y2;
            }
        }
    }
#else
    int half = head_dim / 2;
    for (int pos = 0; pos < seq_len; ++pos) {
        int pos_base = pos * head_dim * heads;
        int cosbase  = pos * head_dim;
        for (int h = 0; h < heads; ++h) {
            int base = pos_base + h * head_dim;
            for (int i = 0; i < half; ++i) {
                float x1 = buf[base + i];
                float x2 = buf[base + i + half];
                float cos_low  = cosbuf[cosbase + i];
                float sin_low  = sinbuf[cosbase + i];
                float cos_high = cosbuf[cosbase + i + half];
                float sin_high = sinbuf[cosbase + i + half];
                float y1 = x1 * cos_low - x2 * sin_low;
                float y2 = x2 * cos_high + x1 * sin_high;
                buf[base + i]        = y1;
                buf[base + i + half] = y2;
            }
        }
    }
#endif
}

static inline void apply_rope_packed(vector<float>& buf, const vector<float>& cosbuf, const vector<float>& sinbuf,
                                     int heads, int head_dim, int seq_len) {
#ifdef __AVX512F__
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        const int split_base = h * seq_len * head_dim;
        int pos_end = seq_len / 4;
        for (int pos = 0; pos < pos_end; ++pos) {
            const int pos_base = split_base + pos * 4 * head_dim;
            const int cosbase  = pos * 4 * head_dim;
            int i = 0;
            for (; i + 4 <= half; i += 4) {
                const float* px1 = &buf[pos_base + i * 4];
                const float* px2 = &buf[pos_base + (i + half) * 4];
                const float* pcosl = &cosbuf[cosbase + i * 4];
                const float* psinl = &sinbuf[cosbase + i * 4];
                const float* pcosh = &cosbuf[cosbase + (i + half) * 4];
                const float* psinh = &sinbuf[cosbase + (i + half) * 4];

                __m512 x1       = _mm512_loadu_ps(px1);
                __m512 x2       = _mm512_loadu_ps(px2);
                __m512 cos_low  = _mm512_loadu_ps(pcosl);
                __m512 sin_low  = _mm512_loadu_ps(psinl);
                __m512 cos_high = _mm512_loadu_ps(pcosh);
                __m512 sin_high = _mm512_loadu_ps(psinh);

                __m512 y1 = _mm512_fmsub_ps(x1, cos_low, _mm512_mul_ps(x2, sin_low));
                __m512 y2 = _mm512_fmadd_ps(x2, cos_high, _mm512_mul_ps(x1, sin_high));

                _mm512_storeu_ps(&buf[pos_base + i * 4],         y1);
                _mm512_storeu_ps(&buf[pos_base + (i + half) * 4], y2);
            }
            for (; i < half; ++i) {
                for (int l = 0; l < 4; ++l) {
                    int base = pos_base + l;
                    float x1 = buf[base + i * 4];
                    float x2 = buf[base + (i + half) * 4];
                    float cos_low  = cosbuf[cosbase + i * 4];
                    float sin_low  = sinbuf[cosbase + i * 4];
                    float cos_high = cosbuf[cosbase + (i + half) * 4];
                    float sin_high = sinbuf[cosbase + (i + half) * 4];
                    float y1 = x1 * cos_low - x2 * sin_low;
                    float y2 = x2 * cos_high + x1 * sin_high;
                    buf[base + i * 4]         = y1;
                    buf[base + (i + half) * 4] = y2;
                }
            }
        }
    }
#else
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        const int split_base = h * seq_len * head_dim;
        int pos_end = seq_len / 4;
        for (int pos = 0; pos < pos_end; ++pos) {
            const int pos_base = split_base + pos * 4 * head_dim;
            for (int i = 0; i < half; ++i) {
                for (int l = 0; l < 4; ++l) {
                    int base = pos_base + l;
                    const int cosbase  = pos * 4 * head_dim + l;
                    float x1 = buf[base + i * 4];
                    float x2 = buf[base + (i + half) * 4];
                    float cos_low  = cosbuf[cosbase + i * 4];
                    float sin_low  = sinbuf[cosbase + i * 4];
                    float cos_high = cosbuf[cosbase + (i + half) * 4];
                    float sin_high = sinbuf[cosbase + (i + half) * 4];
                    float y1 = x1 * cos_low - x2 * sin_low;
                    float y2 = x2 * cos_high + x1 * sin_high;
                    buf[base + i * 4]         = y1;
                    buf[base + (i + half) * 4] = y2;
                }
            }
        }
    }
#endif
}

static inline vector<float> multi_head_attention_blas(const vector<float>& X,
                                                 const vector<float>& Wq,
                                                 const vector<float>& Wk,
                                                 const vector<float>& Wv,
                                                 const vector<float>& Wo,
                                                 const vector<float>& sin_cache,
                                                 const vector<float>& cos_cache,
                                                 const vector<float>& sint_cache,
                                                 const vector<float>& cost_cache,
                                                 const vector<float>& mask,
                                                 const vector<float>& ref_q,
                                                 const vector<float>& ref_rope,
                                                 const vector<float>& ref_attn_softmax,
                                                 int seq_len,
                                                 int emb_dim,
                                                 int q_heads, int kv_heads) {
    int q_head_dim = emb_dim / q_heads;
    int kv_head_dim = seq_len / kv_heads;
    int head_ratio = q_heads / kv_heads;
    int beta = 0;
    int alpha = 1;
    vector<float> Q(emb_dim * seq_len, 0.0f);
    vector<float> K(seq_len * seq_len, 0.0f);
    vector<float> V(seq_len * seq_len, 0.0f);
    vector<float> out_heads(seq_len * emb_dim, 0.0f);
    vector<float> attn(seq_len * seq_len, 0.0f);

    gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasMid,
                seq_len, emb_dim, emb_dim,
                alpha,
                X.data(), emb_dim,
                Wq.data(), emb_dim,
                beta,
                Q.data(), q_head_dim,
                q_head_dim, 0);
    gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasPos,
                seq_len, seq_len, emb_dim,
                alpha,
                X.data(), emb_dim,
                Wk.data(), emb_dim,
                beta,
                K.data(), seq_len,
                -1, 0);
    gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasPos,
                seq_len, seq_len, emb_dim,
                alpha,
                X.data(), emb_dim,
                Wv.data(), emb_dim,
                beta,
                V.data(), seq_len,
                -1, 0);

    // --- RoPE using cos_cache and sin_cache ---
    // clock_t rope_s = clock();

    apply_rope_packed(Q, cost_cache, sint_cache, q_heads, q_head_dim, seq_len);
    // printf("ROPE-Packed time: %.2f ms\n", 1000.0 * (double)(clock() - rope_s) / CLOCKS_PER_SEC);
    // rope_s = clock();
    // apply_rope(Q, cos_cache, sin_cache, q_heads, q_head_dim, seq_len);
    // printf("ROPE-Unpacked time: %.2f ms\n", 1000.0 * (double)(clock() - rope_s) / CLOCKS_PER_SEC);
    // FILE *file_nai_c = fopen("q_rope_packed.csv","w");
    // for(int i=0; i<(seq_len*emb_dim)/4; i++){
    //     for (int j=0; j<4; j++){
    //         if(j>0)
    //             fprintf(file_nai_c, ",");
    //         fprintf(file_nai_c, "%.8f", Q[i*4 + j]);
    //     }
    //     fprintf(file_nai_c, "\n");
    // }
    apply_rope(K, cos_cache, sin_cache, kv_heads, kv_head_dim, seq_len);

    // reshape and duplicate KV heads
    // vector<float> K_rep = repeat_kv_heads(K, seq_len, kv_heads, kv_head_dim, head_ratio);
    // vector<float> V_rep = repeat_kv_heads(V, seq_len, kv_heads, kv_head_dim, head_ratio);

    // attention scores: [seq_len, seq_len]
    int out_addr = 0;
    for (int h = 0; h < q_heads; ++h) {
        gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasMid,
            seq_len, seq_len, q_head_dim,
            alpha,
            Q.data() + h * q_head_dim * seq_len, emb_dim,
            K.data() + h/head_ratio * q_head_dim, seq_len,
            beta,
            attn.data(), seq_len,
            seq_len/2, 0);
        // FILE *file_nai_c = fopen("attn.csv","w");
        // for(int i=0; i<seq_len; i++){
        //     for (int j=0; j<seq_len; j++){
        //         if(j>0)
        //             fprintf(file_nai_c, ",");
        //         fprintf(file_nai_c, "%.8f", attn[i*seq_len + j]);
        //     fprintf(file_nai_c, "\n");
        // }

        // TODO: If mask has different values for each position, this is wrong
        for (int i = 0; i < seq_len * seq_len; ++i){
            attn[i] = attn[i] / sqrtf((float)q_head_dim) + mask[i];
        }
        for (int i = 0; i < seq_len/4; ++i)
            softmax_inplace_packed(attn, i * seq_len/2 * 4, seq_len, q_head_dim, seq_len/2);
        // weighted sum: [seq_len, 64]
        // int t = 4;
        if (h<21){
            out_addr += (h%7 == 0 && h) ? 448*seq_len : 0;
            gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasMid,
                seq_len, q_head_dim, seq_len,
                alpha,
                attn.data(), seq_len,
                V.data() + h/head_ratio * q_head_dim, seq_len,
                beta,
                out_heads.data() + (h%7) * q_head_dim * 4 + out_addr, 1792, //448*4
                -1, 448);
        }else if (h<26){
            out_addr += (h%7 == 0 && h) ? 448*seq_len : 0;
            gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasMid,
                seq_len, q_head_dim, seq_len,
                alpha,
                attn.data(), seq_len,
                V.data() + h/head_ratio * q_head_dim, seq_len,
                1.0f,
                out_heads.data() + (h%7) * q_head_dim * 4 + out_addr, 1408,
                -1, 352);
        } else if (h==26){
            out_addr += (h%7 == 0 && h) ? 448*seq_len : 0;
            gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasMid,
                seq_len, q_head_dim/2, seq_len,
                alpha,
                attn.data(), seq_len,
                V.data() + h/head_ratio * q_head_dim, seq_len,
                beta,
                out_heads.data() + (h%7) * q_head_dim * 4 + out_addr, 1408,
                -1, 352);
            out_addr = seq_len * (q_head_dim/2 + 26 * q_head_dim);
            gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasMid,
                seq_len, q_head_dim/2, seq_len,
                alpha,
                attn.data(), seq_len,
                V.data() + h/head_ratio * q_head_dim + q_head_dim/2, seq_len,
                1.0f,
                out_heads.data() + out_addr, seq_len,
                q_head_dim/2, 352);
            out_addr += q_head_dim * 2;
        } else{
            // FILE *file_nat_c = fopen("out_heads_new.csv","w");;
            // for(int i=0; i<(seq_len*emb_dim); i++){
            //     if((i>0) && (i%4!=0))
            //         fprintf(file_nat_c, ",");
            //     fprintf(file_nat_c, "%.8f", out_heads[i]);
            //     if ((i+1)%4==0){
            //         fprintf(file_nat_c, "\n");
            //     }
            // }
            gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasMid,
                seq_len, q_head_dim, seq_len,
                alpha,
                attn.data(), seq_len,
                V.data() + h/head_ratio * q_head_dim, seq_len,
                1.0f,
                out_heads.data() + out_addr, 1792, //448*4
                -1, 352);
            out_addr += q_head_dim * 4;
        }

        //     }
        // } else{
        //     gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasMid,
        //         seq_len, q_head_dim, seq_len,
        //         alpha,
        //         attn.data(), seq_len,
        //         V.data() + h/head_ratio * q_head_dim, seq_len,
        //         1.0f,
        //         out_heads.data() + (h%7) * q_head_dim * 4 + out_addr, 1792, //448*4
        //         -1, 448);
        // }
        // gemm_seq(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasMid,
        //     seq_len, q_head_dim, seq_len,
        //     alpha,
        //     attn.data(), seq_len,
        //     V.data() + h/head_ratio * q_head_dim, seq_len,
        //     1.0f,
        //     out_heads.data() + h * q_head_dim, emb_dim,
        //     -1, 0);
    }
    // FILE *file_nat_c = fopen("out_heads_new.csv","w");;
    // for(int i=0; i<(seq_len*emb_dim); i++){
    //     if((i>0) && (i%4!=0))
    //         fprintf(file_nat_c, ",");
    //     fprintf(file_nat_c, "%.8f", out_heads[i]);
    //     if ((i+1)%4==0){
    //         fprintf(file_nat_c, "\n");
    //     }
    // }
    vector<float> out(seq_len * emb_dim, 0.0f);
    gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasMid,
            seq_len, emb_dim, emb_dim,
            alpha,
            out_heads.data(), emb_dim,
            Wo.data(), emb_dim,
            beta,
            out.data(), emb_dim,
            -1, 0);
    return out;
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
                                                 int q_heads, int kv_heads) {
    int q_head_dim = emb_dim / q_heads;
    int kv_head_dim = seq_len / kv_heads;
    int head_ratio = q_heads / kv_heads;
    int beta = 0;
    int alpha = 1;
    vector<float> Q(emb_dim * seq_len, 0.0f);
    vector<float> K(seq_len * seq_len, 0.0f);
    vector<float> V(seq_len * seq_len, 0.0f);
    vector<float> out_heads(seq_len * emb_dim, 0.0f);
    vector<float> attn(seq_len * seq_len, 0.0f);

    matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, emb_dim, emb_dim,
                alpha,
                X.data(), emb_dim,
                Wq.data(), emb_dim,
                beta,
                Q.data(), emb_dim);
    matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, seq_len, emb_dim,
                alpha,
                X.data(), emb_dim,
                Wk.data(), emb_dim,
                beta,
                K.data(), seq_len);
    matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, seq_len, emb_dim,
                alpha,
                X.data(), emb_dim,
                Wv.data(), emb_dim,
                beta,
                V.data(), seq_len);

    // --- RoPE using cos_cache and sin_cache ---
    apply_rope(Q, cos_cache, sin_cache, q_heads, q_head_dim, seq_len);
    // FILE *file_nai_c = fopen("q_rope_original.csv","w");
    // for(int i=0; i<seq_len; i++){
    //     for (int j=0; j<emb_dim; j++){
    //         if(j>0)
    //             fprintf(file_nai_c, ",");
    //         fprintf(file_nai_c, "%.8f", Q[i*emb_dim + j]);
    //     }
    //     fprintf(file_nai_c, "\n");
    // }
    apply_rope(K, cos_cache, sin_cache, kv_heads, kv_head_dim, seq_len);

    // reshape and duplicate KV heads
    // vector<float> K_rep = repeat_kv_heads(K, seq_len, kv_heads, kv_head_dim, head_ratio);
    // vector<float> V_rep = repeat_kv_heads(V, seq_len, kv_heads, kv_head_dim, head_ratio);

    // attention scores: [seq_len, seq_len]

    for (int h = 0; h < q_heads; ++h) {
        matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
            seq_len, seq_len, q_head_dim,
            alpha,
            Q.data() + h * q_head_dim, emb_dim,
            K.data() + h/4 * q_head_dim, seq_len,
            beta,
            attn.data(), seq_len);

        for (int i = 0; i < seq_len * seq_len; ++i)
            attn[i] = attn[i] / sqrtf((float)q_head_dim) + mask[i];

        for (int i = 0; i < seq_len; ++i)
            softmax_inplace(attn, i * seq_len, seq_len);

        // weighted sum: [seq_len, 64]
        matmul(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            seq_len, q_head_dim, seq_len,
            alpha,
            attn.data(), seq_len,
            V.data() + h/4 * q_head_dim, seq_len,
            beta,
            out_heads.data() + h * q_head_dim, emb_dim);
    }
    vector<float> out(seq_len * emb_dim, 0.0f);
    matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
            seq_len, emb_dim, emb_dim,
            alpha,
            out_heads.data(), emb_dim,
            Wo.data(), emb_dim,
            beta,
            out.data(), emb_dim);
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
    int alpha = 1;
    int beta = 0;
    clock_t init, end;
    double naive_time_attn = 0, naive_time_mlp = 0;
    double new_time_attn = 0, new_time_mlp = 0;

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
        //cout << "Inferred seq_len=" << seq_len << " emb_dim=" << emb_dim << " layers=" << N << "\n";
    }

    arg_seq_len = (arg_seq_len) ? arg_seq_len : seq_len;

    while (q_heads > 1 && emb_dim % q_heads != 0) --q_heads;
    while (kv_heads > 1 && emb_dim % kv_heads != 0) --kv_heads;

    // Initialize or load input
    vector<float> residual, residual_rp;
    if (have_weights) residual = load_bin(weights_root + "/0/inp.bin");
    else {
        mt19937 rng(42);
        vector<int> tokens = create_random_tokens(seq_len, vocab_size, rng);
        vector<float> emb = create_random_matrix(vocab_size, emb_dim, rng, 0.02f);
        residual = embedding_lookup(tokens, emb, vocab_size, emb_dim);
    }

    vector<float> global_rms_g(emb_dim, 1.0f);
    const float tol = 0.01f;
    float spd_attn = 0.0f, spd_mlp = 0.0f;
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

        if (arg_seq_len != seq_len)
            mask = reshape_tensor(mask, seq_len, arg_seq_len);

        auto sin_rp = copy_changed_layout(sin_cache, arg_seq_len, emb_dim/q_heads);
        auto cos_rp = copy_changed_layout(cos_cache, arg_seq_len, emb_dim/q_heads);
        auto mask_rp = copy_changed_layout(mask, arg_seq_len, arg_seq_len);
        auto q_rp = copy_changed_layout(q_out, arg_seq_len, emb_dim, emb_dim/q_heads);
        auto rope_rp = copy_changed_layout(q_out_rot, arg_seq_len, emb_dim, emb_dim/q_heads);
        auto attn_softmax_rp = copy_changed_layout(attn_weights_softmax_ref, arg_seq_len, arg_seq_len);
        auto inp_rp = copy_changed_layout(inp, arg_seq_len, emb_dim);
        auto rms_g1_rp = copy_changed_layout(rms_g1, 1, emb_dim);
        auto rms_g2_rp = copy_changed_layout(rms_g2, 1, emb_dim);

        int D_mlp = W_gate.size() / emb_dim;
        vector<float> after_attn(arg_seq_len * emb_dim);
        vector<float> gate(arg_seq_len * D_mlp);
        vector<float> up(arg_seq_len * D_mlp);
        vector<float> mlp_out(arg_seq_len * emb_dim);

        residual = (layer) ? residual : inp;
        residual_rp = (layer) ? residual_rp : inp_rp;

        // 1. RMSNorm: inp → attn_inp
        auto attn_inp = rmsnorm(residual, rms_g1, seq_len, emb_dim);

        compare_tensors(attn_inp_ref, attn_inp, "attn_inp", arg_seq_len*emb_dim, tol);

        // 2. Attention: attn_inp → attn_out
        init = clock();
        auto attn_out = multi_head_attention(attn_inp, Wq, Wk, Wv, Wo, sin_cache, cos_cache, mask, arg_seq_len, emb_dim, q_heads, kv_heads);
        end = clock();
        naive_time_attn = (double)(end - init) / CLOCKS_PER_SEC * 1000;
        compare_tensors(attn_out_ref, attn_out, "attn_out", arg_seq_len*emb_dim, tol);

        // 3. Residual add: inp + attn_out
        for (int i = 0; i < arg_seq_len * emb_dim; ++i)
            after_attn[i] = residual[i] + attn_out[i];
        compare_tensors(inp_norm2, after_attn, "after_attn", arg_seq_len*emb_dim, tol);

        // 4. RMSNorm: after_attn → mlp_inp
        auto mlp_inp = rmsnorm(after_attn, rms_g2, arg_seq_len, emb_dim);
        compare_tensors(mlp_inp_ref, mlp_inp, "mlp_inp", arg_seq_len*emb_dim, tol);

        // 5. MLP: mlp_inp → mlp_out
        init = clock();
        matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
            arg_seq_len, D_mlp, emb_dim,
            alpha,
            mlp_inp.data(), emb_dim,
            W_gate.data(), emb_dim,
            beta,
            gate.data(), D_mlp);
        matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
            arg_seq_len, D_mlp, emb_dim,
            alpha,
            mlp_inp.data(), emb_dim,
            W_up.data(), emb_dim,
            beta,
            up.data(), D_mlp);
        compare_tensors(up_proj_out, up, "up_proj_out", arg_seq_len*D_mlp, tol);
        silu_inplace(gate);
        compare_tensors(gate_proj_out, gate, "gate_proj_out", arg_seq_len*D_mlp, tol);
        auto fused = mul(gate, up);
        matmul(CblasRowMajor, CblasNoTrans, CblasTrans,
            arg_seq_len, emb_dim, D_mlp,
            alpha,
            fused.data(), D_mlp,
            W_down.data(), D_mlp,
            beta,
            mlp_out.data(), emb_dim);
        end = clock();
        naive_time_mlp = (double)(init - end) / CLOCKS_PER_SEC * 1000;
        compare_tensors(mlp_out_ref, mlp_out, "mlp_out", arg_seq_len*emb_dim, tol);

        // 6. Residual add: after_attn + mlp_out → out
        vector<float> out(arg_seq_len * emb_dim);
        for (int i = 0; i < arg_seq_len * emb_dim; ++i)
            out[i] = after_attn[i] + mlp_out[i];
        compare_tensors(out_ref, out, "out", arg_seq_len*emb_dim, tol);

        residual = move(out);


        auto transposed_attn_inp = copy_changed_layout(attn_inp, arg_seq_len, emb_dim);
        auto transposed_attn_out = copy_changed_layout(attn_out, arg_seq_len, emb_dim);
        auto transposed_inp_norm2 = copy_changed_layout(inp_norm2, arg_seq_len, emb_dim);
        auto transposed_mlp_inp_ref = copy_changed_layout(mlp_inp_ref, arg_seq_len, emb_dim);
        auto transposed_up_proj_out = copy_changed_layout(up_proj_out, arg_seq_len, D_mlp);
        auto transposed_gate_proj_out = copy_changed_layout(gate_proj_out, arg_seq_len, D_mlp);
        auto transposed_out_ref = copy_changed_layout(out_ref, arg_seq_len, emb_dim);

        // ==========================RP-GEMM==========================
        printf("\n=== RP-GEMM LAYER %d ===\n", layer);

        // 1. RMSNorm: inp → attn_inp
        attn_inp = rmsnorm_packed(residual_rp, rms_g1, arg_seq_len, emb_dim);

        compare_tensors(transposed_attn_inp, attn_inp, "attn_inp", arg_seq_len*emb_dim, tol);

        // 2. Attention: attn_inp → attn_out
        init = clock();
        auto attn_out_rp = multi_head_attention_blas(attn_inp, Wq, Wk, Wv, Wo, sin_cache, cos_cache, sin_rp, cos_rp, mask_rp, q_rp, rope_rp, attn_softmax_rp, arg_seq_len, emb_dim, q_heads, kv_heads);
        end = clock();
        new_time_attn = (double)(end - init) / CLOCKS_PER_SEC * 1000;
        spd_attn += (100*(naive_time_attn-new_time_attn)/naive_time_attn);
        compare_tensors(transposed_attn_out, attn_out_rp, "attn_out", arg_seq_len*emb_dim, tol);
        // 3. Residual add: inp + attn_out
        for (int i = 0; i < arg_seq_len * emb_dim; ++i)
            after_attn[i] = residual_rp[i] + attn_out_rp[i];
        compare_tensors(transposed_inp_norm2, after_attn, "after_attn", arg_seq_len*emb_dim, tol);

        // 4. RMSNorm: after_attn → mlp_inp
        mlp_inp = rmsnorm_packed(after_attn, rms_g2_rp, arg_seq_len, emb_dim);
        compare_tensors(transposed_mlp_inp_ref, mlp_inp, "mlp_inp", arg_seq_len*emb_dim, tol);

        // 5. MLP: mlp_inp → mlp_out
        init = clock();
        gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasMid,
            arg_seq_len, D_mlp, emb_dim,
            alpha,
            mlp_inp.data(), emb_dim,
            W_gate.data(), emb_dim,
            beta,
            gate.data(), D_mlp,
            -1, 0);
        gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasMid,
            arg_seq_len, D_mlp, emb_dim,
            alpha,
            mlp_inp.data(), emb_dim,
            W_up.data(), emb_dim,
            beta,
            up.data(), D_mlp,
            -1, 0);
        compare_tensors(transposed_up_proj_out, up, "up_proj_out", arg_seq_len*D_mlp, tol);
        silu_inplace(gate);
        compare_tensors(transposed_gate_proj_out, gate, "gate_proj_out", arg_seq_len*D_mlp, tol);
        fused = mul(gate, up);
        gemm_seq(CblasRowMajor, CblasNoTrans, CblasTrans, CblasMid,
            arg_seq_len, emb_dim, D_mlp,
            alpha,
            fused.data(), D_mlp,
            W_down.data(), D_mlp,
            beta,
            mlp_out.data(), emb_dim,
            -1, 0);
        end = clock();
        new_time_mlp = (double)(init - end) / CLOCKS_PER_SEC * 1000;
        spd_mlp += (int)(100*(naive_time_mlp-new_time_mlp)/naive_time_mlp);

        // 6. Residual add: after_attn + mlp_out → out
        vector<float> out_rp(arg_seq_len * emb_dim);
        for (int i = 0; i < arg_seq_len * emb_dim; ++i)
            out_rp[i] = after_attn[i] + mlp_out[i];
        compare_tensors(transposed_out_ref, out_rp, "out", arg_seq_len*emb_dim, tol);

        residual_rp = move(out_rp);
    }
    printf("Speedup (Attention): %d%\n", (int)(spd_attn/N));
    printf("Speedup (MLP): %d%\n", (int)(spd_mlp/N));
    cout << "\nAll layers validated.\n";
    return 0;
}
