%% Variable Metric ANALP (VM-ANALP) for L0-CDL 降噪微调主脚本 (动态硬阈值)
% =========================================================================
% 特性:
% 1. 无缝加载 L1 降噪阶段的保存结果进行热启动 (Hot-Start)
% 2. 梯度与数据保真项均针对 y_noisy_cells (带噪图像) 计算
% 3. 实时监控 True-PSNR 以防止 L0 阶段对噪声的过拟合
% 4. 采用 Gerschgorin 变度量矩阵加速 L0 动态硬阈值优化
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 核心参数设置
fprintf('--- 1. 参数设置 (VM-ANALP L0 降噪微调) ---\n');
L1_RESULTS_FILE = 'vmanalp_l1_denoising_results.mat'; % L1 降噪保存的文件

% 【核心降噪参数】: lambda0 决定了动态硬阈值的截断力度。
% 如果发现 L0 后图像噪点仍多，增大 lambda0；如果细节丢失严重（色块化），减小 lambda0。
lambda0 = 0.0002;        

m = 100;              
patch_size = 11;      

% L0 微调极易过拟合噪声，因此 MAXITER 建议设置在 50~150 之间即可
MAXITER = 1;       
max_backtrack_iter = 50; 
gamma1 = 0.0001;         
gamma2 = 0.0001;         
tao1 = 1.5;           
tao2 = 1.5;           

%% 2. 加载 L1 热启动数据与预处理
fprintf('--- 2. 加载 L1 热启动数据 ---\n');
if ~exist(L1_RESULTS_FILE, 'file')
    error('未找到 %s 文件。请确认 VM-ANALP L1 降噪脚本已成功运行并保存！', L1_RESULTS_FILE);
end

% 加载包含干净图像、带噪图像以及预训练字典和系数的工作区
load(L1_RESULTS_FILE, 'y_clean_cells', 'y_noisy_cells', 'D', 'A_cells', 'image_sizes', 'N_l_vec');
numpic = length(y_clean_cells);
d = patch_size^2;

fprintf('热启动加载成功！无缝衔接 L1 结果。\n');

% 计算热启动阶段初始的真实去噪 PSNR
initial_true_psnr = zeros(1, numpic);
for l = 1:numpic
    y_l1_recon = mycol2im(D * A_cells{l}, image_sizes{l}, patch_size);
    initial_true_psnr(l) = psnr(y_clean_cells{l}, y_l1_recon);
end
fprintf('L1 阶段最终的 True-PSNR 平均为: %.4f dB\n\n', mean(initial_true_psnr));

%% 3. 初始化变量与缓存
fprintf('--- 3. 初始化变量与记忆缓存 ---\n');
% 初始化历史步长记忆
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

history.total_objective = zeros(1, MAXITER);
history.data_error = zeros(1, MAXITER);
history.l0_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER); 
history.avg_psnr = zeros(1, MAXITER);  % Noisy-PSNR
history.true_psnr = zeros(1, MAXITER); % True-PSNR

%% 4. VM-ANALP 降噪主循环
fprintf('--- 4. 开始 VM-ANALP L0 极速迭代微调 (动态硬阈值) ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: 更新稀疏编码 A (拟合带噪图像)
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
            unique_sizes = [unique_sizes; sz]; %#ok<AGROW>
            unique_S_alpha{end+1} = new_S;     %#ok<AGROW>
            idx = length(unique_S_alpha);
        end
        S_alpha_cached_list{l} = unique_S_alpha{idx};
    end
    
    new_A_cells = cell(1, numpic);
    
    parfor l = 1:numpic
        old_A_l = A_cells{l}; 
        y_noisy_l = y_noisy_cells{l}; % 【核心】：目标是带噪图像
        img_size_l = image_sizes{l}; 
        
        S_alpha_l = S_alpha_cached_list{l};
        [grad_A_l, ~, H_old] = compute_patch_model_gradients(D, Dt, old_A_l, y_noisy_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        A_l_candidate = old_A_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            
            v_A = old_A_l - grad_A_l ./ S_effective;
            % 动态硬阈值
            lambda_th = sqrt(2 * lambda0 ./ S_effective); 
            A_l_candidate = v_A .* (abs(v_A) > lambda_th);
            
            y_recon_new = mycol2im(D * A_l_candidate, img_size_l, patch_size);
            f_val = 0.5 * sum((y_recon_new - y_noisy_l).^2, 'all');
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
    % Phase 2: 更新字典 D
    % ==========================================================
    [~, grad_D_H_total, H_old_D_update] = compute_H_gradients_in_parallel(D, A_cells, y_noisy_cells, image_sizes, patch_size);
    
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
        if exist('prox_vmetric_L2_ball', 'file')
            D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D);
        else
            D_candidate = proxomiga2_2D(v_D);
        end
        
        H_new_D_update = 0;
        for l = 1:numpic
            y_recon_l = mycol2im(D_candidate * A_cells{l}, image_sizes{l}, patch_size); 
            H_new_D_update = H_new_D_update + 0.5 * sum((y_recon_l - y_noisy_cells{l}).^2, 'all'); 
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
    % Phase 3: 日志与降噪评估
    % ==========================================================
    iter_time = toc;
    if outerIter == 1
        history.time(outerIter) = iter_time;
    else
        history.time(outerIter) = history.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    non_zero_count = sum(cellfun(@nnz, A_cells));
    l0_penalty_term = lambda0 * non_zero_count; 
    total_objective = data_error_term + l0_penalty_term;

    history.total_objective(outerIter) = total_objective;
    history.data_error(outerIter) = data_error_term;
    history.l0_penalty(outerIter) = l0_penalty_term;
    history.sparsity(outerIter) = non_zero_count / (sum(N_l_vec)*m);
    
    % 计算真实的去噪 PSNR
    sum_noisy_psnr = 0;
    sum_true_psnr = 0;
    for l = 1:numpic
        y_recon_l = mycol2im(D * A_cells{l}, image_sizes{l}, patch_size);
        sum_noisy_psnr = sum_noisy_psnr + psnr(y_noisy_cells{l}, y_recon_l);
        sum_true_psnr  = sum_true_psnr  + psnr(y_clean_cells{l}, y_recon_l);
    end
    history.avg_psnr(outerIter) = sum_noisy_psnr / numpic;
    history.true_psnr(outerIter) = sum_true_psnr / numpic;
    
    fprintf('Iter %04d: Obj=%.3e, Noisy-PSNR=%.2f, True-PSNR=%.2f, Sparsity=%.3f%%, L0_Count=%d, Time=%.2fs\n', ...
            outerIter, total_objective, history.avg_psnr(outerIter), history.true_psnr(outerIter), history.sparsity(outerIter)*100, non_zero_count, iter_time);
end

%% 5. 最终可视化评估
fprintf('\n--- 5. 最终降噪评估与可视化 ---\n');
img_idx_to_show = 1;
y_clean_target = y_clean_cells{img_idx_to_show};
y_noisy_input  = y_noisy_cells{img_idx_to_show};

% 从之前的 L1 热启动文件中读取并重建一遍，仅用于对比展示
loaded_data = load(L1_RESULTS_FILE, 'D', 'A_cells'); 
y_l1_recon = mycol2im(loaded_data.D * loaded_data.A_cells{img_idx_to_show}, image_sizes{img_idx_to_show}, patch_size);
y_l0_recon = mycol2im(D * A_cells{img_idx_to_show}, image_sizes{img_idx_to_show}, patch_size);

psnr_noisy = psnr(y_clean_target, y_noisy_input);
psnr_l1    = psnr(y_clean_target, y_l1_recon);
psnr_l0    = psnr(y_clean_target, y_l0_recon);
ssim_l0    = ssim(y_l0_recon,y_clean_target);

% --- 图1：四面板降噪结果对比 ---
figure('Name', 'VM-ANALP L0 - Denoising Comparison', 'Position', [100, 200, 1500, 400]);
subplot(1, 4, 1); imshow(y_clean_target, []); title('原始干净图像');
subplot(1, 4, 2); imshow(y_noisy_input, []); title(sprintf('带噪图像\nPSNR: %.2f dB', psnr_noisy));
subplot(1, 4, 3); imshow(y_l1_recon, []); title(sprintf('L1 粗降噪 (热启动)\nPSNR: %.2f dB', psnr_l1));
subplot(1, 4, 4); imshow(y_l0_recon, []); title(sprintf('VM L0 动态硬阈值微调\nPSNR: %.2f dB', psnr_l0));

% --- 图2：PSNR 演化曲线 ---
figure('Name', 'VM-ANALP L0 - PSNR Evolution');
plot(1:MAXITER, history.avg_psnr, 'r--', 'LineWidth', 1.5); hold on;
plot(1:MAXITER, history.true_psnr, 'g-', 'LineWidth', 2);
yline(mean(initial_true_psnr), 'b:', 'L1 阶段最终 PSNR', 'LineWidth', 1.5);
xlabel('L0 Iteration'); ylabel('平均 PSNR (dB)'); title('VM-ANALP L0 降噪微调 PSNR 演化');
legend('Noisy-PSNR (模型拟合带噪图的程度)', 'True-PSNR (模型真实的降噪能力)', 'Location', 'best');
grid on;

% --- 图3：稀疏度与目标函数下降 ---
figure('Name', 'VM-ANALP L0 - Sparsity Evolution');
yyaxis left;
plot(1:MAXITER, history.sparsity * 100, 'm-', 'LineWidth', 1.5);
ylabel('稀疏度 (%)');
yyaxis right;
plot(1:MAXITER, history.total_objective, 'b-', 'LineWidth', 1.5);
ylabel('目标函数值');
xlabel('迭代次数'); title('稀疏度与目标函数演化');
grid on;