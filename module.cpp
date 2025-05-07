#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

#include <vector>
#include <cmath>
#include <algorithm>

// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX)
{
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val)
{
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ)
{
    // return 0.0;
    return tensor[(x * sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + (z * sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ, float &val)
{
    tensor[(x * sizeX * sizeY * sizeZ) + (y * sizeY * sizeZ) + (z * sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor)
{
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
               twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */

    // -------- YOUR CODE HERE  -------- //

    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < H; h++)
        {

            // Compute Q * K^T for this (b, h)
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++)
                    {
                        float Qval = fourDimRead(Q, b, h, i, k, H, N, d);
                        float Kval = fourDimRead(K, b, h, j, k, H, N, d); // note: K^T means index j here
                        sum += Qval * Kval;
                    }
                    twoDimWrite(QK_t, i, j, N, sum);
                }
            }

            // Softmax row-wise over QK_t
            for (int i = 0; i < N; i++)
            {
                float rowSum = 0.0f;
                for (int j = 0; j < N; j++)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = expf(val);
                    twoDimWrite(QK_t, i, j, N, val);
                    rowSum += val;
                }
                for (int j = 0; j < N; j++)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val /= rowSum;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            // O = softmax(QK^T) * V
            for (int i = 0; i < N; i++)
            {
                for (int k = 0; k < d; k++)
                {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++)
                    {
                        float QKval = twoDimRead(QK_t, i, j, N);
                        float Vval = fourDimRead(V, b, h, j, k, H, N, d);
                        sum += QKval * Vval;
                    }
                    fourDimWrite(O, b, h, i, k, H, N, d, sum);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //
// Utility accessors assumed provided:
// formatTensor, fourDimRead, fourDimWrite, twoDimRead, twoDimWrite

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor,
                                        torch::Tensor KTensor,
                                        torch::Tensor VTensor,
                                        torch::Tensor QK_tTensor,
                                        int B, int H, int N, int d)
{

    // O: (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Flattened buffers
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> QK_t = formatTensor(QK_tTensor); // (N, N)

    const int Tn = 32;
    const int Td = std::min(32, d);

    for (int b = 0; b < B; ++b)
    {
        for (int h = 0; h < H; ++h)
        {

            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    float zero = 0.0f;
                    twoDimWrite(QK_t, i, j, N, zero);
                }
            }

            for (int i0 = 0; i0 < N; i0 += Tn)
            {
                int i_end = std::min(i0 + Tn, N);
                for (int j0 = 0; j0 < N; j0 += Tn)
                {
                    int j_end = std::min(j0 + Tn, N);
                    for (int k0 = 0; k0 < d; k0 += Td)
                    {
                        int k_end = std::min(k0 + Td, d);
                        for (int i = i0; i < i_end; ++i)
                        {
                            for (int j = j0; j < j_end; ++j)
                            {
                                float sum = (k0 == 0)
                                                ? 0.0f
                                                : twoDimRead(QK_t, i, j, N);
                                for (int k2 = k0; k2 < k_end; ++k2)
                                {
                                    float Qv = fourDimRead(Q, b, h, i, k2, H, N, d);
                                    float Kv = fourDimRead(K, b, h, j, k2, H, N, d);
                                    sum += Qv * Kv;
                                }
                                twoDimWrite(QK_t, i, j, N, sum);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < N; ++i)
            {
                float rowSum = 0.0f;
                for (int j = 0; j < N; ++j)
                {
                    float v = twoDimRead(QK_t, i, j, N);
                    v = expf(v);
                    float expv = v;
                    twoDimWrite(QK_t, i, j, N, expv);
                    rowSum += v;
                }
                for (int j = 0; j < N; ++j)
                {
                    float v = twoDimRead(QK_t, i, j, N);
                    float norm = v / rowSum;
                    twoDimWrite(QK_t, i, j, N, norm);
                }
            }

            for (int i0 = 0; i0 < N; i0 += Tn)
            {
                int i_end = std::min(i0 + Tn, N);
                for (int k0 = 0; k0 < d; k0 += Td)
                {
                    int k_end = std::min(k0 + Td, d);
                    for (int j0 = 0; j0 < N; j0 += Tn)
                    {
                        int j_end = std::min(j0 + Tn, N);
                        for (int i = i0; i < i_end; ++i)
                        {
                            for (int j = j0; j < j_end; ++j)
                            {
                                float Pval = twoDimRead(QK_t, i, j, N);
                                for (int k2 = k0; k2 < k_end; ++k2)
                                {
                                    float Vv = fourDimRead(V, b, h, j, k2, H, N, d);
                                    float acc = fourDimRead(O, b, h, i, k2, H, N, d);
                                    acc += Pval * Vv;
                                    float updated = acc;
                                    fourDimWrite(O, b, h, i, k2, H, N, d, updated);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //
inline int index4D(int b, int h, int i, int k, int H, int N, int d)
{
    return ((b * H + h) * N + i) * d + k;
}

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                               int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)

    // Make O Tensor with Shape (B, H, N, d)
    // and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    // Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format ORow Tensor into a 1D vector
    //  You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);

    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    // #pragma omp parallel for collapse(3)
    // loop over batch
    for (int b = 0; b < B; b++)
    {
        // loop over heads
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < N; i++)
            {

                // YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});
                std::vector<float> ORow = formatTensor(ORowTensor);
                // YOUR CODE HERE
                //  Q[i] . K[j]
                for (int j = 0; j < N; j++)
                {
                    float dot = 0.0;
                    for (int k = 0; k < d; k++)
                    {
                        dot += Q[index4D(b, h, i, k, H, N, d)] * K[index4D(b, h, j, k, H, N, d)];
                    }
                    ORow[j] = dot;
                }

                float maxVal = *std::max_element(ORow.begin(), ORow.begin() + N);
                float sum = 0.0;
                for (int j = 0; j < N; j++)
                {
                    ORow[j] = std::exp(ORow[j] - maxVal);
                    sum += ORow[j];
                }

                for (int j = 0; j < N; j++)
                {
                    ORow[j] /= sum;
                }

                // softmax
                for (int k = 0; k < d; k++)
                {
                    float val = 0.0;
                    for (int j = 0; j < N; j++)
                    {
                        val += ORow[j] * V[index4D(b, h, j, k, H, N, d)];
                    }
                    O[index4D(b, h, i, k, H, N, d)] = val;
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
    torch::Tensor /*QiTensor*/, torch::Tensor /*KjTensor*/, torch::Tensor /*VjTensor*/,
    torch::Tensor /*SijTensor*/, torch::Tensor /*PijTensor*/, torch::Tensor /*PVTensor*/,
    torch::Tensor /*OiTensor*/, torch::Tensor /*LTensor*/, torch::Tensor /*LiTensor*/,
    torch::Tensor /*LijTensor*/, torch::Tensor /*LnewTensor*/,
    int Bc, int Br, int B, int H, int N, int d) {

    //Format All Tensors into Vectors
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    
    std::vector<float> Qi(Br * d), Kj(Bc * d), Vj(Bc * d);
    std::vector<float> Sij(Br * Bc), Pij(Br * Bc);
    std::vector<float> PV(Br * d), Oi(Br * d);
    std::vector<float> li(Br), Lij(Br), lnew(Br);
    std::vector<float> PBsum(Br);
    
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; i += Br) {
                
                for (int bi = 0; bi < Br; ++bi) {
                    li[bi]   = -INFINITY;
                    lnew[bi] = 0.0f;
                    for (int k = 0; k < d; ++k) {
                        Oi[bi * d + k] = 0.0f;
                    }
                }

                for (int j = 0; j < N; j += Bc) {
                    
                    for (int bi = 0; bi < Br; ++bi) {
                        int row = i + bi;
                        if (row >= N) break;
                        for (int k = 0; k < d; ++k) {
                            Qi[bi * d + k] = Q[index4D(b, h, row, k, H, N, d)];
                        }
                    }
                    
                    for (int bj = 0; bj < Bc; ++bj) {
                        int col = j + bj;
                        if (col >= N) break;
                        for (int k = 0; k < d; ++k) {
                            Kj[bj * d + k] = K[index4D(b, h, col, k, H, N, d)];
                            Vj[bj * d + k] = V[index4D(b, h, col, k, H, N, d)];
                        }
                    }

                    
                    for (int bi = 0; bi < Br; ++bi) {
                        int row = i + bi;
                        if (row >= N) break;
                        for (int bj = 0; bj < Bc; ++bj) {
                            int col = j + bj;
                            if (col >= N) break;
                            float sum = 0.0f;
                            for (int k = 0; k < d; ++k) {
                                sum += Qi[bi * d + k] * Kj[bj * d + k];
                            }
                            Sij[bi * Bc + bj] = sum;
                        }
                    }

                    
                    for (int bi = 0; bi < Br; ++bi) {
                        int row = i + bi;
                        if (row >= N) break;
                        float m = -INFINITY;
                        for (int bj = 0; bj < Bc; ++bj) {
                            int col = j + bj;
                            if (col >= N) break;
                            m = std::max(m, Sij[bi * Bc + bj]);
                        }
                        Lij[bi] = m;
                    }

                    
                    for (int bi = 0; bi < Br; ++bi) {
                        int row = i + bi;
                        if (row >= N) break;
                        float sumExp = 0.0f;
                        for (int bj = 0; bj < Bc; ++bj) {
                            int col = j + bj;
                            if (col >= N) break;
                            float v = std::exp(Sij[bi * Bc + bj] - Lij[bi]);
                            Pij[bi * Bc + bj] = v;
                            sumExp += v;
                        }
                        PBsum[bi] = sumExp;
                    }

                    
                    for (int bi = 0; bi < Br; ++bi) {
                        int row = i + bi;
                        if (row >= N) break;
                        for (int k = 0; k < d; ++k) {
                            float acc = 0.0f;
                            for (int bj = 0; bj < Bc; ++bj) {
                                int col = j + bj;
                                if (col >= N) break;
                                acc += Pij[bi * Bc + bj] * Vj[bj * d + k];
                            }
                            PV[bi * d + k] = acc;
                        }
                    }

                    
                    for (int bi = 0; bi < Br; ++bi) {
                        int row = i + bi;
                        if (row >= N) break;
                        float old_li = li[bi];
                        float new_li = Lij[bi];
                        float alpha = std::exp(old_li - new_li);

                        
                        lnew[bi] = alpha * lnew[bi] + PBsum[bi];

                       
                        for (int k = 0; k < d; ++k) {
                            Oi[bi * d + k] = alpha * Oi[bi * d + k] + PV[bi * d + k];
                        }

                        
                        li[bi] = new_li;
                    }
                }

              
                for (int bi = 0; bi < Br; ++bi) {
                    int row = i + bi;
                    if (row >= N) break;
                    for (int k = 0; k < d; ++k) {
                        int idx = index4D(b, h, row, k, H, N, d);
                        O[idx] = Oi[bi * d + k] / lnew[bi];
                    }
                }
            }
        }
    }
    

    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
    m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
    m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
    m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
    m.def("twoDimRead", &twoDimRead, "twoDimRead");
    m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
