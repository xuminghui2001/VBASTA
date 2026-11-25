%% Variable Metric ANALP (VM-ANALP) for L1-CDL - 极速降噪优化版
% =========================================================================
% 特性:
% 1. 在自适应度量矩阵 (Variable Metric) 框架下进行 L1 降噪
% 2. 实时监控 Noisy-PSNR 和 True-PSNR 以避免对噪声的过拟合
% 3. 保存降噪结果，以便后续可以进行 L0 微调 (热启动)
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 参数设置
fprintf('--- 1. 参数设置 (VM-ANALP L1 降噪) ---\n');
m = 100;              
patch_size = 11;      
sigma_noise = 25;       % 模拟的噪声标准差 (如 25/255)

% 【降噪核心调参】: L1 正则化参数
% 与纯字典学习不同，降噪时需要将 lambda1 调小，否则图像会过度平滑丢失细节
lambda1 = 0.2;        

% 降噪过程容易过拟合噪声，因此 MAXITER 不需要 1000 次，通常 100~200 次即可早停
MAXITER = 100;       
max_backtrack_iter = 50; 
gamma1 = 0.006;         % S_alpha 的初始度量缩放系数
gamma2 = 0.006;         % S_D 的初始度量缩放系数
tao1 = 1.5;             % A 步长的回溯衰减倍率
tao2 = 1.5;             % D 步长的回溯衰减倍率

%% 2. 加载图像并添加噪声
fprintf('--- 2. 加载干净数据并添加高斯噪声 ---\n');
try 
    L_data = CreateImages('D:\desk\论文\MATLABcdl env\BPG-SPORCO-CDL\bpg_wb_L1\datasets\denoisy','none',0,'gray'); 
    y_clean_cells = squeeze(num2cell(L_data,[1,2,3])); 
catch
    warning('加载失败，使用随机生成的测试数据替代'); 
    for i=1:2, y_clean_cells{i} = rand(100,100); end 
end
numpic = length(y_clean_cells);

y_noisy_cells = cell(1, numpic);
noisy_psnr_initial = zeros(1, numpic);
image_sizes = cell(1, numpic); 
N_l_vec = zeros(1, numpic); 

for l = 1:numpic
    y_clean = y_clean_cells{l};
    % 添加高斯白噪声
    noise = (sigma_noise / 255) * randn(size(y_clean));
    y_noisy = y_clean + noise;
    y_noisy_cells{l} = y_noisy;
    % 计算初始带噪PSNR
    noisy_psnr_initial(l) = psnr(y_clean, y_noisy);
    
    % 预处理尺寸
    image_sizes{l} = [size(y_noisy,1), size(y_noisy,2)]; 
    dummy_patches = myim2col(zeros(image_sizes{l}), patch_size); 
    N_l_vec(l) = size(dummy_patches, 2); 
end
fprintf('初始带噪图像平均 PSNR: %.2f dB\n\n', mean(noisy_psnr_initial));

%% 3. 初始化变量与缓存
fprintf('--- 3. 初始化字典、稀疏码与记忆缓存 ---\n');
d = patch_size^2;
D = randn(d, m);
if exist('proxomiga2_2D', 'file')
    D = proxomiga2_2D(D); 
else
    D = D ./ max(1e-12, vecnorm(D, 2, 1));
end

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
history.time = zeros(1, MAXITER); 
history.avg_psnr = zeros(1, MAXITER);  % 记录模型对噪声图的拟合度
history.true_psnr = zeros(1, MAXITER); % 记录模型真实的降噪能力

%% 4. VM-ANALP 降噪主循环
fprintf('--- 4. 开始 VM-ANALP 极速迭代降噪 ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: 更新稀疏编码 A (注意：全部拟合 y_noisy_cells)
    % ==========================================================
    abs_D = abs(D);
    D_sum = sum(abs_D, 2); 
    Dt = D'; 
    
    % --- 智能缓存 S_alpha ---
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
    
    new_A_cells = cell(1, numpic);
    
    parfor l = 1:numpic
        old_A_l = A_cells{l}; 
        y_noisy_l = y_noisy_cells{l}; 
        img_size_l = image_sizes{l}; 
        
        S_alpha_l = S_alpha_cached_list{l};
        [grad_A_l, ~, H_old] = compute_patch_model_gradients(D, Dt, old_A_l, y_noisy_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        A_l_candidate = old_A_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            v_A = old_A_l - grad_A_l ./ S_effective;
            lambda_th = lambda1 ./ S_effective; 
            A_l_candidate = soft_threshold(v_A, lambda_th);
            
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
    
    old_D = D; 
    tau2_curr = max(gamma2, tau2_history / 1.2);
    
    for k_bt = 1:max_backtrack_iter
        S_effective_D = tau2_curr * S_D;
        v_D = old_D - grad_D_H_total ./ S_effective_D; 
        D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D);
        
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
    l1_penalty_term = lambda1 * sum(abs(cell2mat(A_cells)), 'all');
    total_objective = data_error_term + l1_penalty_term;

    history.total_objective(outerIter) = total_objective;
    history.data_error(outerIter) = data_error_term;
    history.l1_penalty(outerIter) = l1_penalty_term;
    history.sparsity(outerIter) = sum(cellfun(@nnz, A_cells)) / (sum(N_l_vec)*m);
    
    % 计算当前迭代下的 Noisy-PSNR 和 True-PSNR
    sum_noisy_psnr = 0;
    sum_true_psnr = 0;
    for l = 1:numpic
        y_recon_l = mycol2im(D * A_cells{l}, image_sizes{l}, patch_size);
        sum_noisy_psnr = sum_noisy_psnr + psnr(y_noisy_cells{l}, y_recon_l);
        sum_true_psnr  = sum_true_psnr  + psnr(y_clean_cells{l}, y_recon_l);
    end
    history.avg_psnr(outerIter) = sum_noisy_psnr / numpic;
    history.true_psnr(outerIter) = sum_true_psnr / numpic;
    
    fprintf('Iter %04d: Obj=%.2e, Noisy-PSNR=%.2f dB, True-PSNR=%.2f dB, Sparsity=%.2f%%, Time=%.2fs\n', ...
            outerIter, total_objective, history.avg_psnr(outerIter), history.true_psnr(outerIter), history.sparsity(outerIter)*100, iter_time);
end

%% 5. 保存结果供 L0 微调使用
SAVE_FILE = 'vmanalp_l1_denoising_results.mat';
fprintf('\n--- 5. 保存 VM-ANALP 降噪预训练结果到 %s ---\n', SAVE_FILE);
save(SAVE_FILE, 'y_clean_cells', 'y_noisy_cells', 'D', 'A_cells', 'image_sizes', 'N_l_vec', 'history');
fprintf('结果保存完毕。\n\n');

%% 6. 最终可视化评估
fprintf('--- 6. 计算最终评价指标并可视化 ---\n');
img_idx_to_show = 1;
y_clean_target = y_clean_cells{img_idx_to_show};
y_noisy_input  = y_noisy_cells{img_idx_to_show};
A_final = A_cells{img_idx_to_show};
img_size = image_sizes{img_idx_to_show};

y_denoised = mycol2im(D * A_final, img_size, patch_size);
psnr_noisy = psnr(y_clean_target, y_noisy_input);
psnr_denoised = psnr(y_clean_target, y_denoised);
ssim_denoised = ssim(y_denoised, y_clean_target);

% --- 图1：降噪前后图像对比 ---
figure('Name', 'VM-ANALP - Denoising Results', 'Position', [100, 200, 1200, 400]);
subplot(1, 3, 1); imshow(y_clean_target, []); title('原始干净图像');
subplot(1, 3, 2); imshow(y_noisy_input, []); title(sprintf('带噪图像\nPSNR: %.2f dB', psnr_noisy));
subplot(1, 3, 3); imshow(y_denoised, []); title(sprintf('VM-ANALP L1 降噪\nPSNR: %.2f dB', psnr_denoised));

% --- 图2：PSNR 演化曲线 ---
figure('Name', 'VM-ANALP - PSNR Evolution');
plot(1:MAXITER, history.avg_psnr, 'r--', 'LineWidth', 1.5); hold on;
plot(1:MAXITER, history.true_psnr, 'g-', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('平均 PSNR (dB)'); title('VM-ANALP L1 降噪 PSNR 演化');
legend('Noisy-PSNR (模型拟合带噪图的程度)', 'True-PSNR (模型真实的降噪能力)', 'Location', 'best');
grid on;

% --- 图3：学习到的字典 ---
figure('Name', 'VM-ANALP - Learned Dictionary');
if exist('showDictionary', 'file')
    showDictionary(D);
    title(sprintf('基于带噪图像学习到的字典 (m=%d)', m));
else
    imagesc(D); colormap gray; title('学习到的字典矩阵');
end