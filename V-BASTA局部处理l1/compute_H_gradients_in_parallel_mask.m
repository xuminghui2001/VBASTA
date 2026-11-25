function [grad_A_H_cells, grad_D_H_total, total_g_y] = compute_H_gradients_in_parallel_mask(D_L, A_cells, y_masked_cells, M_cells, image_sizes, patch_size)
% COMPUTE_H_GRADIENTS_IN_PARALLEL_MASK 
% 并行计算多张受损图像（带掩码）的 H 梯度和重构误差
%
%   本函数通过 parfor 循环，为元胞数组中的每张图调用带掩码的单图梯度计算函数，
%   并安全、正确地累加关于共享字典 D_L 的梯度，同时计算整体的有效重构误差（仅在 M=1 处）。

    numpic = length(y_masked_cells);
    
    % 初始化用于 parfor 的归约变量 (Reduction Variables)
    grad_D_H_total = zeros(size(D_L));
    grad_A_H_cells = cell(1, numpic);
    total_g_y = 0;

    % 预计算字典转置（在并行循环外计算，避免在各个 worker 中重复运算）
    D_L_t = D_L';
    
    for l = 1:numpic
        % 提取当前图像对应的局部变量，避免广播整个 cell 数组
        A_l = A_cells{l};
        y_masked_l = y_masked_cells{l};
        M_l = M_cells{l};
        img_size_l = image_sizes{l};
        
        % 为第 l 张图调用单图掩码梯度计算函数
        [grad_A_l, grad_D_H_l, g_y_l] = compute_patch_model_gradients_mask(...
            D_L, D_L_t, A_l, y_masked_l, M_l, img_size_l, patch_size);
        
        % 保存当前图像对稀疏系数 A 的梯度
        grad_A_H_cells{l} = grad_A_l;
        
        % 累加字典梯度和重构误差 (parfor 支持这种标准的归约累加)
        grad_D_H_total = grad_D_H_total + grad_D_H_l;
        total_g_y = total_g_y + g_y_l;
    end
end