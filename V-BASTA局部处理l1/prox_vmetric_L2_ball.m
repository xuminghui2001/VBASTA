function D_out = prox_vmetric_L2_ball(V_D, S_D)
% PROX_VMETRIC_L2_BALL: 变度量 (Variable Metric) 下的精确 L2 球体/球面近端映射
% 求解真实的拉格朗日投影优化问题: 
%    arg min_u  0.5 * sum_i S_i * (u_i - v_i)^2
%    s.t.       ||u||_2 <= 1
% 
% 输入:
%   V_D : d x m 矩阵 (经过变度量梯度下降后的无约束中间变量 x)
%   S_D : d x m 矩阵 (对角度量矩阵的对角线元素 a_i)
%
% 输出:
%   D_out : d x m 矩阵 (严格拉格朗日投影后的结果 u)

    [d, m] = size(V_D);
    D_out = V_D; % 默认初始化为无约束状态 (即 mu = 0，如果它本身在单位球内)

    % 1. 判断哪些列（原子）跑到了单位球外部，需要进行投影约束
    norms = vecnorm(V_D, 2, 1);
    active_cols = norms > 1;

    if ~any(active_cols)
        return; % 如果所有原子都在球内，直接返回
    end

    % 提取需要投影的活跃列
    X = V_D(:, active_cols);
    A = S_D(:, active_cols);

    % 2. 向量化的牛顿法 (Newton's Method) 并行求解长期方程
    % 方程: g(mu) = sum_i [ (a_i * x_i) / (a_i + mu) ]^2 - 1 = 0
    % 求解变量 mu (1 x k 的行向量，k 为越界的原子个数)

    % 启发式良好初值：假设 A 是均匀的 (A \approx mean(A))，预估的 mu
    mean_A = mean(A, 1);
    mu = mean_A .* (norms(active_cols) - 1);
    mu = max(mu, 0); % 投影在球面上，mu 必定 > 0

    max_iter = 20; % 牛顿法收敛极快，通常 3~5 次即可
    tol = 1e-7;    % 误差容忍度

    for iter = 1:max_iter
        % 利用隐式扩展 (Implicit Expansion) 构建分母 (d x k)
        A_plus_mu = A + mu; 
        
        % 根据推导公式，计算当前 mu 对应的 u (d x k)
        U = (A .* X) ./ A_plus_mu;
        
        % 函数值 g(mu) = ||u||^2 - 1
        g_mu = sum(U.^2, 1) - 1;
        
        % 检查最大误差是否达到收敛条件
        if max(abs(g_mu)) < tol
            break;
        end
        
        % 根据链式法则求导 g'(mu) = -2 * sum( (a^2 * x^2) / (a + mu)^3 )
        g_prime_mu = -2 * sum( (A.^2 .* X.^2) ./ (A_plus_mu.^3), 1 );
        
        % 牛顿法迭代更新
        step = g_mu ./ g_prime_mu;
        mu = mu - step;
        
        % 安全防线：因为求的是球外点的投影，乘子 mu 绝不可能为负数
        mu = max(mu, 0);
    end

    % 3. 生成最终精确投影的原子并赋值回原矩阵
    A_plus_mu_final = A + mu;
    D_out(:, active_cols) = (A .* X) ./ A_plus_mu_final;

end