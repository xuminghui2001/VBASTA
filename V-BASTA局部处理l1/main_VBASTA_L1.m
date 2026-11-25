%% Variable Metric ANALP (VM-ANALP) for L1-CDL - 极速优化版
% =========================================================================
% 特性:
% 1. 采用 Gerschgorin 盖尔果圆界作为自适应度量矩阵 (Variable Metric)
% 2. 深度缓存: 利用图像同尺寸特性提取冗余计算，大幅降低 O(N) 算子调用
% 3. 步长记忆 (Warm Start): 缓存每张图的历史步长，跳过无效回溯
% 4. 输出对齐基线表格，集成详尽的指标汇总与可视化画图
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 参数设置
fprintf('--- 1. 参数设置 (VM-ANALP L1 预训练) ---\n');
m = 100;              
patch_size = 11;      
lambda1 = 0.5;        

% VM-ANALP 特有回溯与缩放参数 (使用更平滑的参数组合)
MAXITER = 1000;       
max_backtrack_iter = 50; 
gamma1 = 0.006;         % S_alpha 的初始度量缩放系数
gamma2 = 0.006;         % S_D 的初始度量缩放系数
tao1 = 1.5;           % A 步长的回溯衰减倍率(精细微调)
tao2 = 1.5;           % D 步长的回溯衰减倍率

%% 2. 加载与预处理
fprintf('--- 2. 加载和预处理数据 ---\n');
try 
    L_data = CreateImages('datasets\city_100_100','local_cn',1,'gray'); 
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

%% 3. 初始化变量与缓存
fprintf('--- 3. 初始化字典、稀疏码与记忆缓存 ---\n');
d = patch_size^2;
D = randn(d, m);
D = proxomiga2_2D(D); 
A_cells = cell(1, numpic);
for l = 1:numpic
    A_cells{l} = zeros(m, N_l_vec(l)); 
end

% 初始化历史步长记忆 (Warm Start)
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% 日志记录分配
history.total_objective = zeros(1, MAXITER);
history.data_error = zeros(1, MAXITER);
history.l1_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER); % 新增时间记录

%% 4. VM-ANALP 主循环
fprintf('--- 4. 开始 VM-ANALP 极速迭代优化 ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: 更新稀疏编码 A (利用 S_alpha 可变度量)
    % ==========================================================
    abs_D = abs(D);
    D_sum = sum(abs_D, 2); 
    Dt = D'; 
    
    % --- 优化核心: 智能缓存 S_alpha ---
    S_alpha_cached_list = cell(1, numpic);
    unique_sizes = [];
    unique_S_alpha = {};
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
            
            v_A = old_A_l - grad_A_l ./ S_effective;
            lambda_th = lambda1 ./ S_effective; 
            A_l_candidate = soft_threshold(v_A, lambda_th);
            
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
    
    % --- 计算字典的可变度量矩阵 S_D ---
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
    % ---------------------------------
    
    old_D = D; 
    tau2_curr = max(gamma2, tau2_history / 1.2);
    
    for k_bt = 1:max_backtrack_iter
        S_effective_D = tau2_curr * S_D;
        
        % v_D = old_D - grad_D_H_total ./ S_effective_D; 
        % D_candidate = proxomiga2_2D(v_D); 
        % === 修改后 ===
        v_D = old_D - grad_D_H_total ./ S_effective_D; 
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
    % Phase 3: 日志与评估 (修改为科学计数法以对齐表格)
    % ==========================================================
    iter_time = toc;
    if outerIter == 1
        history.time(outerIter) = iter_time;
    else
        history.time(outerIter) = history.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    l1_penalty_term = lambda1 * sum(abs(cell2mat(A_cells)), 'all');
    total_objective = data_error_term + l1_penalty_term;

    history.total_objective(outerIter) = total_objective;
    history.data_error(outerIter) = data_error_term;
    history.l1_penalty(outerIter) = l1_penalty_term;
    history.sparsity(outerIter) = sum(cellfun(@nnz, A_cells)) / (sum(N_l_vec)*m);
    
    fprintf('Iter %04d: Obj=%.3e (Data:%.3e, L1:%.3e), Sparsity=%.3f%%, Time=%.2fs\n', ...
            outerIter, total_objective, data_error_term, l1_penalty_term, history.sparsity(outerIter)*100, history.time(outerIter));
end

%% 5. 保存结果
SAVE_FILE = 'vmanalp_l1_results.mat';
fprintf('\n--- 5. 保存 VM-ANALP 预训练结果到 %s ---\n', SAVE_FILE);
save(SAVE_FILE, 'D', 'A_cells', 'history');
fprintf('结果保存完毕。\n\n');

%% 6. 最终评估与可视化
fprintf('--- 6. 计算最终评价指标并可视化 ---\n');
sum_psnr = 0;
sum_ssim = 0;
for i = 1:numpic
    y_recon_i = mycol2im(D * A_cells{i}, image_sizes{i}, patch_size);
    sum_psnr = sum_psnr + psnr(y_cells{i}, y_recon_i);
    sum_ssim = sum_ssim + ssim(y_recon_i, y_cells{i});
end
avg_psnr = sum_psnr / numpic;
avg_ssim = sum_ssim / numpic;
% 打印总结面板，完美对齐论文数据表格
fprintf('======================================================\n');
fprintf('最终结果汇总 (%d次迭代):\n', MAXITER);
fprintf('Total Objective (Obj):       %.3e\n', history.total_objective(end));
fprintf('Data Consistent (Data):      %.3e\n', history.data_error(end));
fprintf('Regularization (L1):         %.3e\n', history.l1_penalty(end));
fprintf('Sparsity:                    %.3f%%\n', history.sparsity(end)*100);
fprintf('Total Time (s):              %.3f\n', history.time(end));
fprintf('Average-PSNR (dB):           %.2f\n', avg_psnr);
fprintf('Average-SSIM :               %.2f\n', avg_ssim);
fprintf('======================================================\n\n');

% --- 图1：详细收敛曲线 ---
figure('Name', 'VM-ANALP - Detailed Convergence Curves');
plot(1:MAXITER, history.total_objective, 'b-', 'LineWidth', 2);
hold on;
plot(1:MAXITER, history.data_error, 'r--');
plot(1:MAXITER, history.l1_penalty, 'g:');
hold off;
xlabel('迭代次数'); ylabel('目标函数值'); title('VM-ANALP 详细收敛曲线');
legend('总目标函数', '数据误差项', 'L1惩罚项');
grid on; set(gca, 'YScale', 'log');

% --- 图2：稀疏度演化曲线 ---
figure('Name', 'VM-ANALP - Sparsity Evolution');
plot(1:MAXITER, history.sparsity * 100, 'm-', 'LineWidth', 1.5);
xlabel('迭代次数'); ylabel('稀疏度 (%)'); title('VM-ANALP 稀疏度演化曲线');
grid on;

% --- 图3：图像重建、原始图与残差图对比 ---
img_idx_to_show = 1;
y_original = y_cells{img_idx_to_show};
A_final = A_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};
y_reconstructed = mycol2im(D * A_final, img_size, patch_size);
residual_image = abs(y_original - y_reconstructed);
psnr_val = psnr(y_original, y_reconstructed);

figure('Name', 'VM-ANALP - Image Reconstruction Details');
subplot(1, 3, 1); imshow(y_original, []); title('原始图像');
subplot(1, 3, 2); imshow(y_reconstructed, []); title(sprintf('重建图像 (PSNR: %.2f dB)', psnr_val));
subplot(1, 3, 3); imshow(residual_image, []); title('残差绝对值图像'); colorbar;

% --- 图4：学习到的字典 ---
figure('Name', 'VM-ANALP - Learned Dictionary');
% 检查是否存在 showDictionary 函数
if exist('showDictionary', 'file')
    showDictionary(D);
    title(sprintf('VM-ANALP 学习到的字典 (m=%d, patch\\_size=%d)', m, patch_size));
else
    warning('未找到 showDictionary 函数，跳过字典显示。');
end

% --- 图5：第一张图的稀疏系数矩阵可视化 ---
figure('Name', 'VM-ANALP - Sparse Code Map');
imagesc(abs(A_final));
xlabel('图像块索引 (Patch Index)'); ylabel('字典原子索引 (Atom Index)');
title('第一张图的稀疏系数绝对值');
colorbar;