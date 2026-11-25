function result = compute_descent_lemma_upper_bound_VM(x_old, x_new, grad_H_at_x_old, S_metric)
% COMPUTE_DESCENT_LEMMA_UPPER_BOUND_VM: 计算可变度量下的下降引理右侧上界 Q
% Q(x_new, x_old) = H(x_old) + <x_new - x_old, grad_H> + (1/2) * ||x_new - x_old||_{S_metric}^2
%
% 输入:
%   x_old:           旧的迭代点
%   x_new:           新的候选点
%   grad_H_at_x_old: 在旧点处的梯度
%   S_metric:        当前迭代步的可变度量矩阵 (与 x_old 维度相同)

    delta_x = x_new - x_old;
    
    % 1. 计算内积项 <x_new - x_old, grad_H>
    inner_product = sum(delta_x .* grad_H_at_x_old, 'all');
    
    % 2. 计算加权L2范数平方 (利用 Hadamard 乘积): sum( S .* (delta_x)^2 )
    norm_squared_weighted = sum(S_metric .* (delta_x.^2), 'all');
    
    % 注意：此处返回的只是差值部分，调用时加上 H(x_old) 即可
    result = inner_product + 0.5 * norm_squared_weighted;
end