function D_out = proxomiga2_2D(D_in)
% proxomiga2_2D: 对2D字典矩阵的每一列（原子）进行强制L2归一化
% 确保 D_out 的每一列 d_m 都满足 ||d_m||_2 = 1 (除非 d_m 是零向量)。
%
%   D_in 是一个 d x m 的矩阵

    % 1. 计算每一列（原子）的L2范数
    % vecnorm(D_in, 2, 1) 会返回一个 1 x m 的行向量，包含每一列的范数
    norms = vecnorm(D_in, 2, 1);
    
    % 2. 安全处理：防止除以零
    % 将范数为零的位置替换为1，这样零向量除以1后仍为零向量
    norms(norms == 0) = 1;
    
    % 3. 执行归一化操作
    % 每一列都除以其自身的范数。MATLAB的广播机制会自动处理。
    D_out = D_in ./ norms;

end