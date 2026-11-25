function y = soft_threshold(x, threshold)
% SOFT_THRESHOLD 实现软阈值操作
% y = sign(x) .* max(abs(x) - threshold, 0)
    
    if threshold < 0
        error('阈值必须为非负数');
    end
    
    y = sign(x) .* max(abs(x) - threshold, 0);
end