function [grad_A_H_cells, grad_D_H_total, total_g_y] = compute_H_gradients_in_parallel(D_L, A_cells, y_cells, image_sizes, patch_size)
% COMPUTE_H_GRADIENTS_IN_PARALLEL_NEW 并行计算多张图的H梯度和重构误差
%
%   本函数通过 parfor 循环，为元胞数组中的每张图调用单图梯度计算函数，
%   并正确地累加关于共享字典 D_L 的梯度，同时计算总重构误差。

    numpic = length(y_cells);
    grad_D_H_total = zeros(size(D_L));
    grad_A_H_cells = cell(1, numpic);
    total_g_y = 0;

    % 预计算字典转置（在并行循环外）
    D_L_t = D_L';
    
    for l = 1:numpic
        % 为第 l 张图调用单图梯度计算函数
        [grad_A_H_cells{l}, grad_D_H_l, g_y_l] = compute_patch_model_gradients(D_L, D_L_t, A_cells{l}, y_cells{l}, image_sizes{l}, patch_size);
        
        % 累加字典梯度和重构误差
        grad_D_H_total = grad_D_H_total + grad_D_H_l;
        total_g_y = total_g_y + g_y_l;
    end
end