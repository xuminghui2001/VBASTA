%% Variable Metric ANALP (VM-ANALP) for L0 Image Inpainting (Dynamic Hard Thresholding)
% =========================================================================
% 特性:
% 1. 严格热启动 (Strict Hot-Start): 直接继承 L1 阶段的 Mask、受损图和字典。
% 2. 变度量 L0 近端算子: 结合盖尔果圆界实现空间感知的动态硬阈值截断。
% 3. 填表级对齐: 输出严谨的 Target Objective (Data Error + ||A||_0) 以供对比。
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 参数设置与环境加载
fprintf('--- 1. 准备环境 (VM-ANALP L0 Inpainting 阶段) ---\n');

% 指定 L1 预训练生成的文件
L1_RESULTS_FILE = 'vmanalp_l1_inpainting_results.mat'
% 'vmanalp_l1_inpainting_results.mat'; 

if ~exist(L1_RESULTS_FILE, 'file')
    error('未找到 L1 预训练文件 %s。请务必先运行 L1 脚本以生成一致的随机掩码和受损数据！', L1_RESULTS_FILE);
end

fprintf('正在加载 L1 预训练数据并实施严格热启动...\n');
% 直接载入所有变量：D, A_cells, y_clean_cells, y_masked_cells, M_cells, image_sizes, N_l_vec
load(L1_RESULTS_FILE, 'D', 'A_cells', 'y_clean_cells', 'y_masked_cells', 'M_cells', 'image_sizes', 'N_l_vec');

[d, m] = size(D);
patch_size = sqrt(d);
numpic = length(y_clean_cells);

fprintf('成功加载环境，已锁定相同的 Mask (50%%丢失) 和受损图状态！\n\n');

%% 2. L0 优化特定参数
fprintf('--- 2. L0 动态硬阈值参数初始化 ---\n');

% 【关键参数】 内部优化驱动参数 lambda0
% 用于控制变度量下的硬阈值截断强度
lambda0 = 0.00001;        

% 优化步长与回溯参数
MAXITER = 1;       % 热启动下收敛较快
max_backtrack_iter = 50; 
gamma1 = 0.0001;         
gamma2 = 0.0001;         
tao1 = 1.5;           
tao2 = 1.5;           

tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% 分配 L0 阶段专用的日志数组
history_L0.total_objective = zeros(1, MAXITER);
history_L0.data_error = zeros(1, MAXITER);
history_L0.l0_penalty = zeros(1, MAXITER);
history_L0.sparsity = zeros(1, MAXITER);
history_L0.time = zeros(1, MAXITER); 
history_L0.target_objective = zeros(1, MAXITER); 
history_L0.psnr = zeros(1, MAXITER);

%% 3. VM-ANALP Inpainting 主循环 (L0 动态硬阈值)
fprintf('--- 3. 开始 VM-ANALP L0 Inpainting 极速迭代优化 ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: 更新稀疏编码 A (VM 梯度下降 + L0 动态硬阈值 + Mask)
    % ==========================================================
    abs_D = abs(D); D_sum = sum(abs_D, 2); Dt = D'; 
    
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
    
    new_A_cells = cell(1, numpic);
    
    parfor l = 1:numpic
        old_A_l = A_cells{l}; 
        y_masked_l = y_masked_cells{l}; 
        M_l = M_cells{l};
        img_size_l = image_sizes{l}; 
        S_alpha_l = S_alpha_cached_list{l};
        
        % 1. 计算带 Mask 的梯度
        [grad_A_l, ~, H_old] = compute_patch_model_gradients_mask(D, Dt, old_A_l, y_masked_l, M_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        A_l_candidate = old_A_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            v_A = old_A_l - grad_A_l ./ S_effective;
            
            % 2. 变度量下的 L0 动态硬阈值
            lambda_th = sqrt(2 * lambda0 ./ S_effective); 
            A_l_candidate = v_A .* (abs(v_A) > lambda_th);
            
            % 3. 带掩码的下降引理检验
            y_recon_new = mycol2im(D * A_l_candidate, img_size_l, patch_size);
            f_val = 0.5 * sum((M_l .* (y_recon_new - y_masked_l)).^2, 'all');
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
    % Phase 2: 更新字典 D (变度量球投影 + Mask)
    % ==========================================================
    [~, grad_D_H_total, H_old_D_update] = compute_H_gradients_in_parallel_mask(D, A_cells, y_masked_cells, M_cells, image_sizes, patch_size);
    
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
        D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D); % L2 精确投影
        
        H_new_D_update = 0;
        for l = 1:numpic
            y_recon_l = mycol2im(D_candidate * A_cells{l}, image_sizes{l}, patch_size); 
            % 带掩码的残差计算
            H_new_D_update = H_new_D_update + 0.5 * sum((M_cells{l} .* (y_recon_l - y_masked_cells{l})).^2, 'all'); 
        end
        
        Q_val = H_old_D_update + compute_descent_lemma_upper_bound_VM(old_D, D_candidate, grad_D_H_total, S_effective_D);
        if H_new_D_update <= Q_val
            tau2_history = tau2_curr; 
            break; 
        else
            tau2_curr = tau2_curr * tao2; 
        end
    end
    D = D_candidate; 
    
    % ==========================================================
    % Phase 3: 日志与评估
    % ==========================================================
    iter_time = toc;
    if outerIter == 1
        history_L0.time(outerIter) = iter_time;
    else
        history_L0.time(outerIter) = history_L0.time(outerIter-1) + iter_time;
    end
    
    data_error_term = H_new_D_update; 
    non_zero_count = sum(cellfun(@nnz, A_cells)); 
    
    l0_penalty_term = lambda0 * non_zero_count; 
    total_objective = data_error_term + l0_penalty_term;
    target_objective = data_error_term + non_zero_count;

    history_L0.total_objective(outerIter) = total_objective;
    history_L0.target_objective(outerIter) = target_objective; 
    history_L0.data_error(outerIter) = data_error_term;
    history_L0.l0_penalty(outerIter) = l0_penalty_term;
    history_L0.sparsity(outerIter) = non_zero_count / (sum(N_l_vec)*m);
    
    % 计算 Inpainting 修复 PSNR (已知 + 预测)
    current_psnr = 0;
    for l = 1:numpic
        y_recon_curr = mycol2im(D * A_cells{l}, image_sizes{l}, patch_size);
        y_final_curr = M_cells{l} .* y_clean_cells{l} + (1 - M_cells{l}) .* y_recon_curr;
        current_psnr = current_psnr + psnr(y_clean_cells{l}, y_final_curr);
    end
    history_L0.psnr(outerIter) = current_psnr / numpic;
    
    fprintf('Iter %04d: TargetObj=%.3e [Opt=%.3e], L0_Count:%d, PSNR=%.2fdB, Sparsity=%.3f%%, Time=%.2fs\n', ...
            outerIter, target_objective, total_objective, non_zero_count, history_L0.psnr(outerIter), history_L0.sparsity(outerIter)*100, history_L0.time(outerIter));
end

%% 4. 最终评估与可视化
fprintf('\n--- 4. 最终评价指标汇总 ---\n');

fprintf('======================================================\n');
fprintf('L0 修复任务最终结果汇总 (%d次迭代):\n', MAXITER);
fprintf('Obj for Table (Data + ||A||_0):   %.3e  <-- 填入论文表格\n', history_L0.target_objective(end));
fprintf('------------------------------------------------------\n');
fprintf('Non-Zero Elements Count:          %d\n', non_zero_count);
fprintf('Sparsity:                         %.3f%%\n', history_L0.sparsity(end)*100);
fprintf('Total Time (s):                   %.3f\n', history_L0.time(end));
fprintf('Average-PSNR (dB):                %.2f\n', history_L0.psnr(end));
fprintf('======================================================\n\n');

% --- 图1：收敛与非零个数演化 ---
figure('Name', 'VM-ANALP L0 Inpainting - Convergence');
subplot(2,1,1);
plot(1:MAXITER, history_L0.target_objective, 'b-', 'LineWidth', 2); hold on;
plot(1:MAXITER, history_L0.data_error, 'r--'); 
plot(1:MAXITER, history_L0.target_objective - history_L0.data_error, 'g:'); hold off;
xlabel('迭代次数'); ylabel('值'); title('目标函数收敛曲线 (Data + L0_Count)');
legend('Target Obj', 'Data Error', 'L0 Count');
grid on; set(gca, 'YScale', 'log');

subplot(2,1,2);
plot(1:MAXITER, history_L0.psnr, 'm-', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('PSNR (dB)'); title('图像修复 PSNR 演化');
grid on;

% --- 图2：图像重建视觉对比 ---
img_idx_to_show = 1;
y_original = y_clean_cells{img_idx_to_show};
y_masked = y_masked_cells{img_idx_to_show};
M_mask = M_cells{img_idx_to_show};
A_final = A_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};

y_reconstructed = mycol2im(D * A_final, img_size, patch_size);
y_final = M_mask .* y_original + (1 - M_mask) .* y_reconstructed;
final_psnr = psnr(y_original, y_final);
final_ssim = ssim(y_final,y_original);
figure('Name', 'VM-ANALP L0 Inpainting - Visual Results', 'Position', [100, 100, 1200, 400]);
subplot(1, 4, 1); imshow(y_original, []); title('原始图像');
subplot(1, 4, 2); imshow(y_masked, []); title('退化图像 (50% 丢失)');
subplot(1, 4, 3); imshow(y_reconstructed, []); title('纯字典 L0 重建');
subplot(1, 4, 4); imshow(y_final, []); title(sprintf('最终修复融合结果\n(PSNR: %.2f dB)', final_psnr));