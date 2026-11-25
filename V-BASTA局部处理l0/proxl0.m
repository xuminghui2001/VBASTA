% function result = proxl0(x, eta, lambda)
% % 函数功能：计算向量x关于\(\ell_0\)范数的近端算子
% % 输入参数：
% % x：输入向量
% % eta：公式中的\(\eta\)参数
% % lambda：公式中的\(\lambda\)参数
% % 输出参数：
% % result：计算得到的近端算子结果向量
% 
% threshold = sqrt(2 * eta * lambda); % 计算阈值
% n = length(x); % 获取向量长度
% result = zeros(size(x)); % 初始化结果向量
% for i = 1:n
%     if abs(x(i)) > threshold
%         result(i) = x(i);
%     elseif abs(x(i)) == threshold
%         result(i) = 0;
%     else
%         result(i) = 0;
%     end
% end
% end
function result = proxl0(x, eta, lambda)
    % 函数功能：计算向量x关于\(\ell_0\)范数的近端算子
    % 输入参数：
    % x：输入向量
    % eta：公式中的\(\eta\)参数
    % lambda：公式中的\(\lambda\)参数
    % 输出参数：
    % result：计算得到的近端算子结果向量
    threshold = sqrt(2 * eta * lambda); % 计算阈值
    % 使用逻辑索引进行向量化操作
    result = x .* (abs(x) > threshold);
end