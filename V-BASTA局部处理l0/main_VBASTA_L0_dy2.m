%% Variable Metric ANALP (VM-ANALP) for L0-CDL (Dynamic Hard Thresholding)
% =========================================================================
% 特性:
% 1. L1 到 L0 的平滑热启动 (Hot-Start)
% 2. 采用 Gerschgorin 变度量矩阵加速梯度下降
% 3. 严格使用变度量下的 L0 近端算子 (动态硬阈值)，实现空间感知截断
% 4. 包含 S_alpha 的极速维度安全缓存机制
% 5. 【核心展示】双目标函数记录与对比 (优化驱动Obj vs 论文填表Obj)
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 参数设置
fprintf('--- 1. 参数设置 (VM-ANALP L0 阶段 - 动态硬阈值版) ---\n');
USE_HOT_START = true;%记得确认这里
L1_RESULTS_FILE = 'D:\desk\论文\MATLABcdl env\VMANALP_EX\ablation\robust\L1\change gamma\l11001105gamma=0.1.mat'; % L1 预训练保存的文件

m = 100;              
patch_size = 11;      

% 【关键参数】 内部优化驱动参数 lambda0
% 用于控制实际的截断阈值，调整此参数以达到你预期的稀疏度 (如 0.066%)
lambda0 = 0.5;%0.3 0.4 0.5   

% VM-ANALP 特有回溯与缩放参数 (已根据你的最佳实践调整)
MAXITER = 1000;       
max_backtrack_iter = 50; 
gamma1 = 0.1;         
gamma2 = 0.1;         
tao1 = 1.5;           
tao2 = 1.5;           

%% 2. 加载与预处理
fprintf('--- 2. 加载和预处理数据 ---\n');
try 
    L_data = CreateImages('datasets/city_100_100','local_cn',1,'gray'); %local_cn
    y_cells = squeeze(num2cell(L_data,[1,2,3])); 
catch
    warning('加载失败，使用随机生成的测试数据替代'); 
    for i=1:2, y_cells{i} = rand(100,100); end 
end
numpic = length(y_cells);

image_sizes = cell(1, numpic); 
N_l_vec = zeros(1, numpic); 
for l = 1:numpic
    y_l = y_cells{l}; 
    image_sizes{l} = [size(y_l,1), size(y_l,2)]; 
    dummy_patches = myim2col(zeros(image_sizes{l}), patch_size); 
    N_l_vec(l) = size(dummy_patches, 2); 
end
fprintf('预处理完成。共加载 %d 张图像。\n\n', numpic);

%% 3. 初始化变量 (稳健的热启动逻辑)
fprintf('--- 3. 初始化变量 (热启动检测) ---\n');
d = patch_size^2;

if USE_HOT_START && exist(L1_RESULTS_FILE, 'file')
    fprintf('检测到热启动选项，正在从 %s 加载并验证...\n', L1_RESULTS_FILE);
    loaded_data = load(L1_RESULTS_FILE, 'D', 'A_cells'); 
    is_compatible = true;
    
    if ~(ismatrix(loaded_data.D) && size(loaded_data.D,1)==d && size(loaded_data.D,2)==m)
        warning('加载的字典D维度不匹配！'); is_compatible = false; 
    end
    if is_compatible && ~(iscell(loaded_data.A_cells) && length(loaded_data.A_cells)==numpic)
        warning('加载的稀疏系数A_cells图像数量不匹配！'); is_compatible = false; 
    end
    if is_compatible
        for l=1:numpic
            if size(loaded_data.A_cells{l},2) ~= N_l_vec(l)
                warning('图像 %d 的A_cells块数不匹配！', l); is_compatible = false; break; 
            end
        end
    end
    
    if is_compatible
        D = loaded_data.D; 
        A_cells = loaded_data.A_cells; 
        fprintf('热启动加载成功！无缝衔接 L1 结果。\n\n'); 
    else
        USE_HOT_START = false; warning('热启动验证失败，将采用随机初始化。'); 
    end
else
    if USE_HOT_START, warning('热启动文件 %s 未找到', L1_RESULTS_FILE); end
    USE_HOT_START = false;
end

if ~USE_HOT_START
    fprintf('采用随机初始化...\n\n'); 
    D = randn(d, m); 
    D = prox_vmetric_L2_ball(D, ones(d, m)); % 使用严格球面投影初始化
    A_cells = cell(1, numpic); 
    for l=1:numpic, A_cells{l} = zeros(m, N_l_vec(l)); end
end

% 初始化历史步长记忆
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% history 结构，用于记录双目标函数
history.total_objective = zeros(1, MAXITER);   % 优化算法驱动的 Obj
history.target_objective = zeros(1, MAXITER);  % 论文填表专用的 Obj
history.data_error = zeros(1, MAXITER);
history.l0_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER); 

%% 4. VM-ANALP 主循环
fprintf('--- 4. 开始 VM-ANALP L0 极速迭代优化 (动态硬阈值) ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: 更新稀疏编码 A (VM 梯度下降 + L0 动态硬阈值)
    % ==========================================================
    abs_D = abs(D);
    D_sum = sum(abs_D, 2); 
    Dt = D'; 
    
    % --- 智能缓存 S_alpha ---
    S_alpha_cached_list = cell(1, numpic);
    unique_sizes = []; unique_S_alpha = {};
    for l = 1:numpic
        sz = image_sizes{l};
        idx = [];
        if ~isempty(unique_sizes)
            matches = (unique_sizes(:,1) == sz(1)) & (unique_sizes(:,2) == sz(2));
            idx = find(matches, 1);
        end
        if isempty(idx)
            D_rep = repmat(D_sum, 1, N_l_vec(l));
            Img_overlap = mycol2im(D_rep, sz, patch_size);
            P_overlap = myim2col(Img_overlap, patch_size);
            new_S = abs_D' * P_overlap + 1e-8;
            unique_sizes = [unique_sizes; sz]; 
            unique_S_alpha{end+1} = new_S;     
            idx = length(unique_S_alpha);
        end
        S_alpha_cached_list{l} = unique_S_alpha{idx};
    end
    % ---------------------------------
    
    new_A_cells = cell(1, numpic);
    
    parfor l = 1:numpic
        old_A_l = A_cells{l}; 
        y_l = y_cells{l}; 
        img_size_l = image_sizes{l}; 
        
        S_alpha_l = S_alpha_cached_list{l};
        [grad_A_l, ~, H_old] = compute_patch_model_gradients(D, Dt, old_A_l, y_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        A_l_candidate = old_A_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            
            % 1. VM 梯度下降
            v_A = old_A_l - grad_A_l ./ S_effective;
            
            % 2. 动态硬阈值 (L0 Proximal Mapping under Variable Metric)
            lambda_th = sqrt(2 * lambda0 ./ S_effective); 
            A_l_candidate = v_A .* (abs(v_A) > lambda_th);
            
            % 3. 检查下降引理二次上界
            y_recon_new = mycol2im(D * A_l_candidate, img_size_l, patch_size);
            f_val = 0.5 * sum((y_recon_new - y_l).^2, 'all');
            Q_val = H_old + compute_descent_lemma_upper_bound_VM(old_A_l, A_l_candidate, grad_A_l, S_effective);
            
            if f_val <= Q_val
                tau1_history(l) = tau1_curr; 
                break; 
            else
                tau1_curr = tau1_curr * tao1; 
            end
        end
        new_A_cells{l} = A_l_candidate;
    end
    A_cells = new_A_cells; 
    
    % ==========================================================
    % Phase 2: 更新字典 D (利用 S_D 可变度量)
    % ==========================================================
    [~, grad_D_H_total, H_old_D_update] = compute_H_gradients_in_parallel(D, A_cells, y_cells, image_sizes, patch_size);

    S_D = zeros(d, m);
    for l = 1:numpic
        abs_A_l = abs(A_cells{l});
        A_sum = sum(abs_A_l, 1); 
        X = repmat(A_sum, d, 1); 
        Img_overlap = mycol2im(X, image_sizes{l}, patch_size); 
        P_overlap = myim2col(Img_overlap, patch_size);         
        S_D = S_D + P_overlap * abs_A_l';                      
    end
    S_D = S_D + 1e-8; 

    old_D = D; 
    tau2_curr = max(gamma2, tau2_history / 1.2);

    for k_bt = 1:max_backtrack_iter
        S_effective_D = tau2_curr * S_D;

        v_D = old_D - grad_D_H_total ./ S_effective_D; 
        % 采用你推导的严格的牛顿法变度量球面投影
        D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D);

        H_new_D_update = 0;
        for l = 1:numpic
            y_recon_l = mycol2im(D_candidate * A_cells{l}, image_sizes{l}, patch_size); 
            H_new_D_update = H_new_D_update + 0.5 * sum((y_recon_l - y_cells{l}).^2, 'all'); 
        end
        f_val = H_new_D_update; 
        Q_val = H_old_D_update + compute_descent_lemma_upper_bound_VM(old_D, D_candidate, grad_D_H_total, S_effective_D);

        if f_val <= Q_val
            tau2_history = tau2_curr; 
            break; 
        else
            tau2_curr = tau2_curr * tao2; 
        end
    end
    D = D_candidate; 
    
    % ==========================================================
    % Phase 3: 日志与双目标函数评估
    % ==========================================================
    iter_time = toc;
    if outerIter == 1
        history.time(outerIter) = iter_time;
    else
        history.time(outerIter) = history.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    non_zero_count = sum(cellfun(@nnz, A_cells)); % 统计全图非零元素个数
    
    % 1. 底层真实的优化目标函数 (Data + lambda0 * ||A||_0)
    l0_penalty_term = lambda0 * non_zero_count; 
    total_objective = data_error_term + l0_penalty_term;

    % 2. 填表专用的物理目标函数: Data + 1 * ||A||_0
    target_objective = data_error_term + non_zero_count;

    history.total_objective(outerIter) = total_objective;
    history.target_objective(outerIter) = target_objective; 
    history.data_error(outerIter) = data_error_term;
    history.l0_penalty(outerIter) = l0_penalty_term;
    history.sparsity(outerIter) = non_zero_count / (sum(N_l_vec)*m);
    
    % 控制台输出明确双目标
    fprintf('Iter %04d: Obj(Table)=%.3e | Obj(Opt)=%.3e | Data:%.3e | L0_Count:%d | Sparsity=%.3f%% | Time=%.2fs\n', ...
            outerIter, target_objective, total_objective, data_error_term, non_zero_count, history.sparsity(outerIter)*100, history.time(outerIter));
end

%% 5. 保存结果
SAVE_FILE = 'vmanalp_l0_results.mat';
fprintf('\n--- 5. 保存 VM-ANALP L0 结果到 %s ---\n', SAVE_FILE);
save(SAVE_FILE, 'D', 'A_cells', 'history');
fprintf('结果保存完毕。\n\n');

%% 6. 最终评估与可视化
fprintf('--- 6. 计算最终评价指标并可视化 ---\n');
sum_psnr = 0;
sum_ssim = 0;
for i = 1:numpic
    y_original = y_cells{i};
    y_recon_i = mycol2im(D * A_cells{i}, image_sizes{i}, patch_size);
    
    sum_psnr = sum_psnr + psnr(y_original, y_recon_i);
    sum_ssim = sum_ssim + ssim(y_recon_i, y_original);
end
avg_psnr = sum_psnr / numpic;
avg_ssim = sum_ssim / numpic;

% 打印总结面板
fprintf('======================================================\n');
fprintf('最终 L0 结果汇总 (%d次迭代):\n', MAXITER);
fprintf('Obj for Table (Data + ||A||_0):   %.3e  <-- 请将此数值填入论文表格！\n', history.target_objective(end));
fprintf('Internal Opt Obj (Data + lam*L0): %.3e  <-- 算法内部收敛值\n', history.total_objective(end));
fprintf('------------------------------------------------------\n');
fprintf('Data Consistent (Data Error):     %.3e\n', history.data_error(end));
fprintf('Non-Zero Elements Count:          %d\n', sum(cellfun(@nnz, A_cells)));
fprintf('Sparsity:                         %.3f%%\n', history.sparsity(end)*100);
fprintf('Total Time (s):                   %.3f\n', history.time(end));
fprintf('Average-PSNR (dB):                %.2f\n', avg_psnr);
fprintf('Average-SSIM:                     %.4f\n', avg_ssim);
fprintf('======================================================\n\n');

% =========================================================================
% 画图部分重点：同时展示两个目标函数曲线
% =========================================================================
figure('Name', 'VM-ANALP L0 - Dual Objective Convergence');

% 画出专门用于填表对比的目标函数 (Data + ||A||_0)
plot(1:MAXITER, history.target_objective, 'b-', 'LineWidth', 2.5);
hold on;
% 画出驱动算法优化的实际目标函数 (Data + \lambda * ||A||_0)
plot(1:MAXITER, history.total_objective, 'm-', 'LineWidth', 1.5);
% 画出纯数据误差项 (Data Error)
plot(1:MAXITER, history.data_error, 'r--', 'LineWidth', 1.5);

hold off;
xlabel('迭代次数 (Iterations)', 'FontSize', 11, 'FontWeight', 'bold'); 
ylabel('目标函数值 (Log Scale)', 'FontSize', 11, 'FontWeight', 'bold'); 
title('VM-ANALP (L_0) 目标函数收敛曲线对比', 'FontSize', 12);
% 添加极度清晰的图例
legend('Obj for Table (Data Error + ||\alpha||_0)', ...
       'Internal Obj (Data Error + \lambda ||\alpha||_0)', ...
       'Data Error', 'Location', 'northeast', 'FontSize', 10);
grid on; 
set(gca, 'YScale', 'log', 'FontSize', 10);

% --- 图2：稀疏度演化曲线 ---
figure('Name', 'VM-ANALP L0 - Sparsity Evolution');
plot(1:MAXITER, history.sparsity * 100, 'm-', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('稀疏度 (%)'); title('VM-ANALP L0 稀疏度演化曲线');
grid on;

% --- 图3：图像重建、原始图与残差图对比 ---
img_idx_to_show = 1;
y_original = y_cells{img_idx_to_show};
A_final = A_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};
y_reconstructed = mycol2im(D * A_final, img_size, patch_size);
residual_image = abs(y_original - y_reconstructed);

figure('Name', 'VM-ANALP L0 - Image Reconstruction Details');
subplot(1, 3, 1); imshow(y_original, []); title('原始图像');
subplot(1, 3, 2); imshow(y_reconstructed, []); title(sprintf('重建图像 (PSNR: %.2f dB)', psnr(y_original, y_reconstructed)));
subplot(1, 3, 3); imshow(residual_image, []); title('残差绝对值图像'); colorbar;

% --- 图4：学习到的字典 ---
if exist('showDictionary', 'file')
    figure('Name', 'VM-ANALP L0 - Learned Dictionary');
    showDictionary(D);
    title(sprintf('VM-ANALP L0 学习到的字典 (m=%d, patch\\_size=%d)', m, patch_size));
end