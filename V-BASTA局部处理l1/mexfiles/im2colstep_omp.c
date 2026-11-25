#include "mex.h"
#include <omp.h>

#define X_IN   prhs[0]
#define SZ_IN  prhs[1]
#define S_IN   prhs[2]
#define B_OUT  plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) { 
    double *x, *b, *s;
    mwSize sz[3], stepsize[3], n[3], ndims;
    
    if (nrhs < 2 || nrhs > 3) mexErrMsgTxt("Invalid number of input arguments."); 
    ndims = mxGetNumberOfDimensions(X_IN);
    
    s = mxGetPr(SZ_IN);
    sz[0] = (mwSize)(s[0] + 0.01); 
    sz[1] = (mwSize)(s[1] + 0.01); 
    sz[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    
    if (nrhs == 3) {
        s = mxGetPr(S_IN);
        stepsize[0] = (mwSize)(s[0] + 0.01); 
        stepsize[1] = (mwSize)(s[1] + 0.01); 
        stepsize[2] = ndims==3 ? (mwSize)(s[2] + 0.01) : 1;
    } else {
        stepsize[0] = stepsize[1] = stepsize[2] = 1;
    }
    
    n[0] = (mxGetDimensions(X_IN))[0]; 
    n[1] = (mxGetDimensions(X_IN))[1]; 
    n[2] = ndims==3 ? (mxGetDimensions(X_IN))[2] : 1;
    
    mwSize num_blocks_i = (n[0]-sz[0])/stepsize[0] + 1;
    mwSize num_blocks_j = (n[1]-sz[1])/stepsize[1] + 1;
    mwSize num_blocks_k = (n[2]-sz[2])/stepsize[2] + 1;
    mwSize total_blocks = num_blocks_i * num_blocks_j * num_blocks_k;
    mwSize block_len = sz[0]*sz[1]*sz[2];
    
    /* ====== 核心优化：分配未初始化内存，零额外开销 ====== */
    B_OUT = mxCreateUninitNumericMatrix(block_len, total_blocks, mxDOUBLE_CLASS, mxREAL);
    
    x = mxGetPr(X_IN);
    b = mxGetPr(B_OUT);
            
    /* ====== 核心优化：使用 collapse(3) 彻底干掉除法和取余 ====== */
    #pragma omp parallel for collapse(3) schedule(static)
    for (mwIndex k_idx = 0; k_idx < num_blocks_k; k_idx++) {
        for (mwIndex j_idx = 0; j_idx < num_blocks_j; j_idx++) {
            for (mwIndex i_idx = 0; i_idx < num_blocks_i; i_idx++) {
                
                mwIndex i = i_idx * stepsize[0];
                mwIndex j = j_idx * stepsize[1];
                mwIndex k = k_idx * stepsize[2];

                // 仅使用极快的乘法和加法计算一维偏移量
                mwIndex b_idx = k_idx * (num_blocks_i * num_blocks_j) + j_idx * num_blocks_i + i_idx;
                double *b_base = b + b_idx * block_len;

                for (mwIndex m=0; m<sz[2]; m++) {
                    for (mwIndex l=0; l<sz[1]; l++) {
                        double *b_ptr = b_base + m*sz[0]*sz[1] + l*sz[0];
                        double *x_ptr = x + (k+m)*n[0]*n[1] + (j+l)*n[0] + i;
                        
                        /* 强制使用 SIMD 向量化替代 memcpy，实现零分支开销 */
                        #pragma omp simd
                        for (mwIndex t=0; t<sz[0]; t++) {
                            b_ptr[t] = x_ptr[t];
                        }
                    }
                }
            }
        }
    }
}