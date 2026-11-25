%% Variable Metric ANALP (VM-ANALP) for L1 Image Inpainting (Pre-training)
% =========================================================================
% 特性:
% 1. 执行基于 L1 软阈值的修复字典学习预训练。
% 2. 严格生成并锁定随机掩码 (Mask) 和退化图像。
% 3. 将 D, A_cells 以及所有的掩码环境打包保存，为后续 L0 热启动提供无偏环境。
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 参数设置
fprintf('--- 1. 参数设置 (VM-ANALP L1 Inpainting 预训练) ---\n');
m = 100;              
patch_size = 11;      
lambda1 = 0.04;         % L1 稀疏正则化系数
noiseSD = 0;            % 初始噪声标准差

MAXITER = 300;          % 预训练迭代次数
max_backtrack_iter = 50; 
gamma1 = 0.006;         
gamma2 = 0.006;         
tao1 = 1.5;           
tao2 = 1.5;           

%% 2. 加载数据与生成 50% 丢失掩码
fprintf('--- 2. 加载数据并生成 50%% 掩码 ---\n');
try 
    L_data = CreateImages('datasets/standard1','none',0,'gray'); 
    y_clean_cells = squeeze(num2cell(L_data,[1,2,3])); 
catch
    warning('加载失败，使用随机生成的测试数据替代'); 
    for i=1:2, y_clean_cells{i} = rand(100,100); end 
end
numpic = length(y_clean_cells);

M_cells = cell(1, numpic);
y_masked_cells = cell(1, numpic);
image_sizes = cell(1, numpic); 
N_l_vec = zeros(1, numpic); 

% 生成掩码与受损图像 (关键：这些数据将被保存供 L0 使用)
for l = 1:numpic
    sz = [size(y_clean_cells{l},1), size(y_clean_cells{l},2)];
    image_sizes{l} = sz;
    
    M_cells{l} = zeros(sz);
    M_cells{l}(rand(sz) < 0.5) = 1; % 50% 随机掩码
    
    y_noisy = y_clean_cells{l} + noiseSD * randn(sz);
    y_masked_cells{l} = M_cells{l} .* y_noisy;
    
    dummy_patches = myim2col(zeros(sz), patch_size); 
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

tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

history.total_objective = zeros(1, MAXITER);
history.data_error = zeros(1, MAXITER);
history.l1_penalty = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER);
history.time = zeros(1, MAXITER); 
history.psnr = zeros(1, MAXITER);

%% 4. VM-ANALP Inpainting 主循环 (L1 软阈值)
fprintf('--- 4. 开始 VM-ANALP L1 Inpainting 预训练迭代 ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % Phase 1: 更新稀疏编码 A (VM + L1 软阈值 + Mask)
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
        
        [grad_A_l, ~, H_old] = compute_patch_model_gradients_mask(D, Dt, old_A_l, y_masked_l, M_l, img_size_l, patch_size);
        
        tau1_curr = max(gamma1, tau1_history(l) / 1.2); 
        A_l_candidate = old_A_l; 
        
        for k_bt = 1:max_backtrack_iter
            S_effective = tau1_curr * S_alpha_l; 
            v_A = old_A_l - grad_A_l ./ S_effective;
            
            % L1 软阈值收缩
            lambda_th = lambda1 ./ S_effective; 
            A_l_candidate = soft_threshold(v_A, lambda_th);
            
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
        D_candidate = prox_vmetric_L2_ball(v_D, S_effective_D);
        
        H_new_D_update = 0;
        for l = 1:numpic
            y_recon_l = mycol2im(D_candidate * A_cells{l}, image_sizes{l}, patch_size); 
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
    
    current_psnr = 0;
    for l = 1:numpic
        y_recon_curr = mycol2im(D * A_cells{l}, image_sizes{l}, patch_size);
        y_final_curr = M_cells{l} .* y_clean_cells{l} + (1 - M_cells{l}) .* y_recon_curr;
        current_psnr = current_psnr + psnr(y_clean_cells{l}, y_final_curr);
    end
    history.psnr(outerIter) = current_psnr / numpic;
    
    fprintf('Iter %04d: Obj=%.3e, Sparsity=%.3f%%, PSNR=%.2fdB, Time=%.2fs\n', ...
            outerIter, total_objective, history.sparsity(outerIter)*100, history.psnr(outerIter), history.time(outerIter));
end

%% 5. 保存结果 (关键步骤)
SAVE_FILE = 'vmanalp_l1_inpainting_results.mat';
fprintf('\n--- 5. 打包保存 L1 预训练结果与掩码环境到 %s ---\n', SAVE_FILE);
% 严格打包：除了优化的变量，掩码和图也要保存以备 L0 热启动读取
save(SAVE_FILE, 'D', 'A_cells', 'history', ...
     'y_clean_cells', 'y_masked_cells', 'M_cells', 'image_sizes', 'N_l_vec');
fprintf('结果保存完毕。L0 脚本可直接读取此文件进行严谨对比。\n\n');

%% 6. 最终评估与可视化
fprintf('--- 6. L1 预训练最终效果展示 ---\n');
figure('Name', 'VM-ANALP L1 Inpainting - Convergence');
plot(1:MAXITER, history.total_objective, 'b-', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('总目标函数值'); title('L1 预训练收敛曲线');
grid on; set(gca, 'YScale', 'log');

img_idx_to_show = 1;
y_original = y_clean_cells{img_idx_to_show};
y_masked = y_masked_cells{img_idx_to_show};
M_mask = M_cells{img_idx_to_show};
A_final = A_cells{img_idx_to_show};

y_reconstructed = mycol2im(D * A_final, image_sizes{img_idx_to_show}, patch_size);
y_final = M_mask .* y_original + (1 - M_mask) .* y_reconstructed;
final_psnr = psnr(y_original, y_final);

figure('Name', 'VM-ANALP L1 Inpainting - Visual Results', 'Position', [100, 100, 1200, 400]);
subplot(1, 4, 1); imshow(y_original, []); title('原始图像');
subplot(1, 4, 2); imshow(y_masked, []); title('退化图像 (50% 丢失)');
subplot(1, 4, 3); imshow(y_reconstructed, []); title('纯字典 L1 重建');
subplot(1, 4, 4); imshow(y_final, []); title(sprintf('L1 修复结果\n(PSNR: %.2f dB)', final_psnr));