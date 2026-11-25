#include "mex.h"
#include <omp.h>
#include <stdlib.h>

/* 定义自适应内存安全阈值 (当前设为 64MB，可根据你的硬件 L3 缓存调整) */
#define MAX_SAFE_MEMORY_BYTES (64 * 1024 * 1024)

#define B_IN   prhs[0]
#define N_IN   prhs[1]
#define SZ_IN  prhs[2]
#define S_IN   prhs[3]
#define X_OUT  plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) { 
    double *x, *b, *s;
    mwSize sz[3], stepsize[3], n[3], ndims;
    
    if (nrhs < 3 || nrhs > 4) mexErrMsgTxt("Invalid number of input arguments."); 
    ndims = mxGetM(N_IN)*mxGetN(N_IN);
    
    s = mxGetPr(N_IN);
    n[0] = (mwSize)(s[0] + 0.01); 
    n[1] = (mwSize)(s[1] + 0.01); 
    n[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    s = mxGetPr(SZ_IN);
    sz[0] = (mwSize)(s[0] + 0.01); 
    sz[1] = (mwSize)(s[1] + 0.01); 
    sz[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    if (nrhs == 4) {
        s = mxGetPr(S_IN);
        stepsize[0] = (mwSize)(s[0] + 0.01); 
        stepsize[1] = (mwSize)(s[1] + 0.01); 
        stepsize[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    } else {
        stepsize[0] = stepsize[1] = stepsize[2] = 1;
    }
    
    mwSize num_blocks_i = (n[0]-sz[0])/stepsize[0] + 1;
    mwSize num_blocks_j = (n[1]-sz[1])/stepsize[1] + 1;
    mwSize num_blocks_k = (n[2]-sz[2])/stepsize[2] + 1;
    mwSize block_len = sz[0]*sz[1]*sz[2];
    mwSize total_img_size = n[0]*n[1]*n[2];
    
    X_OUT = mxCreateNumericArray(ndims, n, mxDOUBLE_CLASS, mxREAL);
    b = mxGetPr(B_IN);
    x = mxGetPr(X_OUT);
            
    int num_threads = omp_get_max_threads();
    
    /* ====== 核心：动态探测内存需求 ====== */
    size_t required_memory = (size_t)num_threads * (size_t)total_img_size * sizeof(double);
    
    if (required_memory <= MAX_SAFE_MEMORY_BYTES) {
        /* ----------------------------------------------------
           策略 A：数据量小 -> 走 Map-Reduce 无锁高速公路 
        ---------------------------------------------------- */
        double **x_locals = (double**)malloc(num_threads * sizeof(double*));
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            x_locals[tid] = (double*)calloc(total_img_size, sizeof(double));
            double *my_x = x_locals[tid];

            #pragma omp for collapse(3) schedule(static)
            for (mwIndex k_idx = 0; k_idx < num_blocks_k; k_idx++) {
                for (mwIndex j_idx = 0; j_idx < num_blocks_j; j_idx++) {
                    for (mwIndex i_idx = 0; i_idx < num_blocks_i; i_idx++) {
                        mwIndex i = i_idx * stepsize[0];
                        mwIndex j = j_idx * stepsize[1];
                        mwIndex k = k_idx * stepsize[2];
                        
                        mwIndex b_idx = k_idx * (num_blocks_i * num_blocks_j) + j_idx * num_blocks_i + i_idx;
                        double *b_base = b + b_idx * block_len;

                        for (mwIndex m=0; m<sz[2]; m++) {
                            for (mwIndex l=0; l<sz[1]; l++) {
                                double *b_ptr = b_base + m*sz[0]*sz[1] + l*sz[0];
                                mwIndex x_offset = (k+m)*n[0]*n[1] + (j+l)*n[0] + i;
                                #pragma omp simd
                                for (mwIndex t=0; t<sz[0]; t++) {
                                    my_x[x_offset + t] += b_ptr[t];
                                }
                            }
                        }
                    }
                }
            }
        }

        /* Map-Reduce 最终汇总 */
        #pragma omp parallel for schedule(static)
        for (mwIndex p = 0; p < total_img_size; p++) {
            double sum = 0.0;
            for (int t = 0; t < num_threads; t++) {
                sum += x_locals[t][p];
            }
            x[p] += sum;
        }

        for (int t = 0; t < num_threads; t++) free(x_locals[t]);
        free(x_locals);

    } else {
        /* ----------------------------------------------------
           策略 B：数据超大 -> 走 Atomic 安全降级，零额外内存
        ---------------------------------------------------- */
        #pragma omp parallel for collapse(3) schedule(static)
        for (mwIndex k_idx = 0; k_idx < num_blocks_k; k_idx++) {
            for (mwIndex j_idx = 0; j_idx < num_blocks_j; j_idx++) {
                for (mwIndex i_idx = 0; i_idx < num_blocks_i; i_idx++) {
                    mwIndex i = i_idx * stepsize[0];
                    mwIndex j = j_idx * stepsize[1];
                    mwIndex k = k_idx * stepsize[2];
                    
                    mwIndex b_idx = k_idx * (num_blocks_i * num_blocks_j) + j_idx * num_blocks_i + i_idx;
                    double *b_base = b + b_idx * block_len;

                    for (mwIndex m=0; m<sz[2]; m++) {
                        for (mwIndex l=0; l<sz[1]; l++) {
                            double *b_ptr = b_base + m*sz[0]*sz[1] + l*sz[0];
                            mwIndex x_offset = (k+m)*n[0]*n[1] + (j+l)*n[0] + i;
                            
                            // 放弃 simd，逐个像素进行轻量级原子累加
                            for (mwIndex t=0; t<sz[0]; t++) {
                                #pragma omp atomic
                                x[x_offset + t] += b_ptr[t];
                            }
                        }
                    }
                }
            }
        }
    }
}