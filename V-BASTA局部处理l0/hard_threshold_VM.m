function x_thresh = hard_threshold_VM(x, k, S_metric)
% HARD_THRESHOLD_VM: 基于可变度量空间 (Variable Metric) 的硬约束投影
% 目标: \min_{\|x\|_0 \le k} \frac{1}{2} \|x - v\|_{S_metric}^2
% 
% 输入：
%   x        - 输入矩阵/张量 (即梯度下降后的 v_A)
%   k        - 保留的最大非零元素数
%   S_metric - 对应位置的可变度量矩阵 (如 S_effective)
% 输出：
%   x_thresh - 在可变度量空间下截断后的结果

    % 安全检查并确定 k 值
    x_flat = x(:);
    S_flat = S_metric(:);
    k = max(min(floor(k), numel(x_flat)), 0); 
    
    if k == 0
        x_thresh = zeros(size(x), 'like', x); 
        return;
    end
    
    % 【统一量纲】：计算度量加权绝对值 (避免平方带来的极值)
    energy_flat = sqrt(S_flat) .* abs(x_flat);
    
    if k == numel(x_flat) 
        threshold = min(energy_flat);
    else
        % 找到第 k 大的"能量绝对值"
        threshold_vals = maxk(energy_flat, k);
        threshold = threshold_vals(end);
    end
    
    % 生成 mask：必须使用同样的 sqrt(S) * abs(x) 进行比较！
    energy_matrix = sqrt(S_metric) .* abs(x);
    x_thresh = x .* (energy_matrix >= threshold); 
end