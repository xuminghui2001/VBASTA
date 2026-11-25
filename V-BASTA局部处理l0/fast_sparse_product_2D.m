%% 一维情况
% Faster version of sparse product that only returns the norm
% function norm_sq = fast_sparse_product(i, j, n)
%     % Compute just the norm squared without constructing the matrix
%     overlap_start = max(i, j);
%     overlap_end = min(i+n-1, j+n-1);
%     norm_sq = max(0, overlap_end - overlap_start + 1);
% end
function norm_sq = fast_sparse_product_2D(i1, j1, i2, j2, ND)
    % Computes the squared norm of the overlap between two patches
    % at positions (i1,j1) and (i2,j2)
    
    % Calculate the overlapping region
    start_i = max(i1, i2);
    end_i = min(i1+ND-1, i2+ND-1);
    start_j = max(j1, j2);
    end_j = min(j1+ND-1, j2+ND-1);
    
    % If no overlap
    if start_i > end_i || start_j > end_j
        norm_sq = 0;
        return;
    end
    
    % Calculate the area of overlap
    overlap_rows = end_i - start_i + 1;
    overlap_cols = end_j - start_j + 1;
    norm_sq = overlap_rows * overlap_cols;
end