%% Cartoon-Texture Separation using VM-ANALP Framework (L0 Version)
% =========================================================================
% 版本: V2.0 (VM-ANALP L0 热启动版)
% 特性:
% 1. 采用 L0 范数约束 (硬阈值 + Top-K 截断) 替代 L1 软阈值，追求极致稀疏
% 2. 引入热启动 (Warm Start)：从预训练的 VM-ANALP L1 结果初始化 D 和 A
% 3. 独立追踪卡通和纹理的最高PSNR，支持连续 patience 次无改进早停
% 4. 四窗口实时监控：分离图像、学习字典、PSNR收敛曲线、稀疏度演化
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 参数设置
fprintf('--- 1. 参数设置 (VM-ANALP L0 分离版本) ---\n');
m = 100;              
patch_size = 11;      

% --- L0 模型参数 ---
lambda0_penalty = 0.001; % L0稀疏项权重 (比L1小，因为硬阈值条件不同)
sparsity_k = 10;         % 每个稀疏编码向量(图像块)的最大非零元素数
zeta = 0.015;            % TV项权重 (控制卡通平滑度，L0下通常可略微降低)

% --- VM-ANALP 优化参数 ---
MAXITER = 300;           % L0 热启动后收敛极快，总迭代次数可调小
max_backtrack_iter = 30; 
gamma1 = 0.006;          % S_alpha 初始缩放
gamma2 = 0.006;          % S_D 初始缩放
tao_bt = 1.5;            % 回溯衰减倍率

% --- 可视化与智能停止参数 ---
plot_interval = 5; 
patience = 30;     

%% 2. 加载数据
fprintf('--- 2. 加载数据 ---\n');
load("datasets\separation\cat4.mat"); 
load("datasets\separation\fense2.mat"); 

I_true = im2double(imgc); 
M_true = 0.4 * im2double(imresize(fense2, [256, 256]));
y = I_true + M_true; 
numpic = 1; 

image_sizes{1} = size(y);
dummy_patches = myim2col(zeros(image_sizes{1}), patch_size);
N_l_vec(1) = size(dummy_patches, 2);

%% 3. 初始化变量 (从 L1 结果热启动)
fprintf('--- 3. 初始化变量 (热启动) ---\n');
d = patch_size^2;

try
    fprintf('正在加载L1预训练结果 (best_separation_results_vmanalp.mat)...\n');
    load('best_separation_results_vmanalp.mat', 'best_D_for_texture', 'best_A_for_texture');
    D = best_D_for_texture;
    A_cells = best_A_for_texture;
    fprintf('成功加载最优 D 和 A_cells。\n');
catch
    warning('未找到L1预训练结果。将回退到随机初始化！(建议先运行L1脚本)');
    D = randn(d, m); D = proxomiga2_2D(D);
    A_cells{1} = zeros(m, N_l_vec(1));
end

% 初始化卡通分量 y_C (基于热启动的纹理直接计算初始残差)
y_T_init = mycol2im(D * A_cells{1}, image_sizes{1}, patch_size);
y_C = y - y_T_init; 

% 步长记忆缓存
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% 最优结果追踪变量
best_psnr_cartoon = -Inf; best_y_C = [];
best_psnr_texture = -Inf; best_D_for_texture = []; best_A_for_texture = {};
iterations_without_improvement = 0;

history.psnr_total = zeros(1, MAXITER);
history.psnr_cartoon = zeros(1, MAXITER);
history.psnr_texture = zeros(1, MAXITER);
history.sparsity = zeros(1, MAXITER); % 新增：稀疏度记录
sumtime = 0;

%% 4a. 实时可视化设置
fprintf('--- 4a. 初始化可视化窗口 ---\n');
hFig1 = figure('Name', 'VM-ANALP L0 实时分离结果', 'Position', [50, 400, 1600, 400]);
hFig2 = figure('Name', 'VM-ANALP L0 实时学习字典', 'Position', [50, 50, 600, 600]);
hFig3 = figure('Name', 'VM-ANALP L0 实时PSNR曲线', 'Position', [700, 50, 600, 400]);
hFig4 = figure('Name', 'VM-ANALP L0 稀疏性监控', 'Position', [1350, 50, 500, 400]);

%% 4b. VM-ANALP (L0) + TV 交替优化主循环
fprintf('--- 4b. VM-ANALP (L0) + TV 交替优化主循环开始 ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % STEP 1: 更新纹理分量 (D, A) - L0 逻辑
    % ==========================================================
    y_target = y - y_C; 
    
    % --- 子步 A: 更新系数 A (L0 硬阈值 + TopK) ---
    abs_D = abs(D); D_sum = sum(abs_D, 2);
    S_alpha = abs_D' * myim2col(mycol2im(repmat(D_sum, 1, N_l_vec(1)), image_sizes{1}, patch_size), patch_size) + 1e-8; 
    
    [grad_A, ~, H_old] = compute_patch_model_gradients(D, D', A_cells{1}, y_target, image_sizes{1}, patch_size);
    tau1_curr = max(gamma1, tau1_history(1) / 1.2);
    
    for k_bt = 1:max_backtrack_iter
        S_effective = tau1_curr * S_alpha;
        v_A = A_cells{1} - grad_A ./ S_effective;
        
        % 【核心修改】VM 框架下的 L0 Proximal Mapping
        A_candidate = v_A;
        % 1. 基于 lambda0 的硬阈值 (Hard Thresholding)
        ht_thresh = sqrt(2 * lambda0_penalty ./ S_effective);
        A_candidate(abs(A_candidate) < ht_thresh) = 0;
        
        % 2. 基于 sparsity_k 的每列保留最大K个非零元
        if sparsity_k < m
            sorted_A = sort(abs(A_candidate), 1, 'descend');
            k_thresh = sorted_A(sparsity_k, :); % 每列第K大的值
            A_candidate(abs(A_candidate) < k_thresh) = 0; % 小于阈值的清零
        end
        
        % 下降引理检查 (针对平滑项的二次上界)
        y_recon_T = mycol2im(D * A_candidate, image_sizes{1}, patch_size);
        f_val = 0.5 * sum((y_recon_T - y_target).^2, 'all');
        Q_val = H_old + compute_descent_lemma_upper_bound_VM(A_cells{1}, A_candidate, grad_A, S_effective);
        
        if f_val <= Q_val
            tau1_history(1) = tau1_curr; break;
        else
            tau1_curr = tau1_curr * tao_bt;
        end
    end
    A_cells{1} = A_candidate;
    
    % --- 记录当前稀疏度 ---
    sparsity_ratio = nnz(A_cells{1}) / numel(A_cells{1});
    history.sparsity(outerIter) = sparsity_ratio;
    
    % --- 子步 B: 更新字典 D (保持不变) ---
    [~, grad_D, H_old_D] = compute_H_gradients_in_parallel(D, A_cells, {y_target}, image_sizes, patch_size);
    
    abs_A = abs(A_cells{1});
    S_D = myim2col(mycol2im(repmat(sum(abs_A, 1), d, 1), image_sizes{1}, patch_size), patch_size) * abs_A' + 1e-8;
    
    tau2_curr = max(gamma2, tau2_history / 1.2);
    for k_bt = 1:max_backtrack_iter
        S_eff_D = tau2_curr * S_D;
        D_candidate = prox_vmetric_L2_ball(D - grad_D ./ S_eff_D, S_eff_D); 
        
        y_recon_T_new = mycol2im(D_candidate * A_cells{1}, image_sizes{1}, patch_size);
        f_val = 0.5 * sum((y_recon_T_new - y_target).^2, 'all');
        Q_val = H_old_D + compute_descent_lemma_upper_bound_VM(D, D_candidate, grad_D, S_eff_D);
        
        if f_val <= Q_val
            tau2_history = tau2_curr; break;
        else
            tau2_curr = tau2_curr * tao_bt;
        end
    end
    D = D_candidate;
    
    % ==========================================================
    % STEP 2: 更新卡通分量 y_C (TV Denoising)
    % ==========================================================
    y_T_reconstructed = mycol2im(D * A_cells{1}, image_sizes{1}, patch_size);
    residual_image = y - y_T_reconstructed;
    
    opts_tv.method = 'l2'; opts_tv.max_itr = 100; opts_tv.tol = 1e-10; opts_tv.rho_r = 1; opts_tv.beta = [1 1 0];
    mu_tv = 1 / zeta;
    tv_out = deconvtv(residual_image, 1, mu_tv, opts_tv);
    y_C = tv_out.f;
    
    % ==========================================================
    % STEP 3: 日志记录、最优追踪与智能停止
    % ==========================================================
    elapsed_time = toc; sumtime = sumtime + elapsed_time;
    reconstructed_img = y_T_reconstructed + y_C;
    
    psnr_total = psnr(reconstructed_img, y);
    psnr_texture = psnr(y_T_reconstructed, M_true);
    psnr_cartoon = psnr(y_C, I_true);
    
    history.psnr_total(outerIter) = psnr_total;
    history.psnr_texture(outerIter) = psnr_texture;
    history.psnr_cartoon(outerIter) = psnr_cartoon;
    
    fprintf('Iter %03d/%d: Total=%.2fdB, Txt=%.2fdB, Crt=%.2fdB | 稀疏度=%.2f%%, 耗时=%.2fs\n', ...
            outerIter, MAXITER, psnr_total, psnr_texture, psnr_cartoon, sparsity_ratio*100, elapsed_time);
            
    % --- 最优结果追踪 ---
    improved = false;
    if psnr_cartoon > best_psnr_cartoon
        best_psnr_cartoon = psnr_cartoon; best_y_C = y_C; improved = true;
        fprintf('       *** 新的最优卡通 (PSNR: %.4f dB) ***\n', best_psnr_cartoon);
    end
    if psnr_texture > best_psnr_texture
        best_psnr_texture = psnr_texture; best_D_for_texture = D; best_A_for_texture = A_cells; improved = true;
        fprintf('       *** 新的最优纹理 (PSNR: %.4f dB) ***\n', best_psnr_texture);
    end
    
    if improved, iterations_without_improvement = 0; 
    else, iterations_without_improvement = iterations_without_improvement + 1; 
    end
    
    if iterations_without_improvement >= patience
        fprintf('\n--- 连续 %d 次迭代无改进，触发 L0 早停。 ---\n', patience);
        history.psnr_total = history.psnr_total(1:outerIter); history.psnr_texture = history.psnr_texture(1:outerIter);
        history.psnr_cartoon = history.psnr_cartoon(1:outerIter); history.sparsity = history.sparsity(1:outerIter);
        break; 
    end

    % ==========================================================
    % STEP 4: 实时可视化更新
    % ==========================================================
    if mod(outerIter, plot_interval) == 0 || outerIter == 1 || outerIter == MAXITER
        if ~isgraphics(hFig1, 'figure'), hFig1 = figure('Name', 'VM-ANALP L0 实时分离结果', 'Position', [50, 400, 1600, 400]); end
        figure(hFig1); subplot(1, 4, 1); imshow(y, []); title('原始图像 (y)'); subplot(1, 4, 2); imshow(y_C, []); title(sprintf('卡通分量 (y_C)\nPSNR: %.2f dB', psnr_cartoon)); subplot(1, 4, 3); imshow(y_T_reconstructed, []); title(sprintf('纹理分量 (y_T)\nPSNR: %.2f dB', psnr_texture)); subplot(1, 4, 4); imshow(reconstructed_img, []); title(sprintf('重构图像 (y_T+y_C)\nPSNR: %.2f dB', psnr_total));
        
        if ~isgraphics(hFig2, 'figure'), hFig2 = figure('Name', 'VM-ANALP L0 实时学习字典', 'Position', [50, 50, 600, 600]); end
        figure(hFig2); showDictionary(D); title(sprintf('VM-ANALP L0 学习字典 (Iter %d)', outerIter));
        
        if ~isgraphics(hFig3, 'figure'), hFig3 = figure('Name', 'VM-ANALP L0 实时PSNR曲线', 'Position', [700, 50, 600, 400]); end
        figure(hFig3); plot(1:outerIter, history.psnr_total(1:outerIter), 'b-', 'DisplayName', 'PSNR(总)'); hold on; plot(1:outerIter, history.psnr_texture(1:outerIter), 'r--', 'DisplayName', 'PSNR(纹理)'); plot(1:outerIter, history.psnr_cartoon(1:outerIter), 'g:', 'DisplayName', 'PSNR(卡通)'); hold off; xlabel('迭代次数'); ylabel('PSNR (dB)'); title('VM-ANALP L0 收敛曲线'); legend show; grid on; xlim([0, MAXITER]);
        
        if ~isgraphics(hFig4, 'figure'), hFig4 = figure('Name', 'VM-ANALP L0 稀疏性监控', 'Position', [1350, 50, 500, 400]); end
        figure(hFig4); plot(1:outerIter, history.sparsity(1:outerIter)*100, 'm-', 'LineWidth', 2); xlabel('迭代次数'); ylabel('稀疏度 (%)'); title('L0稀疏性监控'); grid on; xlim([0, MAXITER]); ylim([0, 100 * sparsity_k / m * 1.5]);
        
        drawnow;
    end
end
fprintf('\n--- VM-ANALP L0 分离完成，总耗时: %.2f 秒 ---\n', sumtime);

%% 5. 保存并展示最优分离结果
fprintf('--- 5. 保存并展示追踪到的最优结果 ---\n');
best_y_T = mycol2im(best_D_for_texture * best_A_for_texture{1}, image_sizes{1}, patch_size);
reconstructed_best = best_y_C + best_y_T;
psnr_best_total = psnr(reconstructed_best, y);
final_sparsity = nnz(best_A_for_texture{1}) / numel(best_A_for_texture{1});

save('best_L0_separation_results_vmanalp.mat', ...
     'best_y_C', 'best_y_T', 'best_D_for_texture', 'best_A_for_texture', ...
     'best_psnr_cartoon', 'best_psnr_texture', 'psnr_best_total', 'final_sparsity');
fprintf('最优结果已保存到 best_L0_separation_results_vmanalp.mat。\n');
fprintf('最终 L0 结果: 卡通 PSNR = %.4f dB, 纹理 PSNR = %.4f dB | 最终稀疏性 = %.2f%%\n', best_psnr_cartoon, best_psnr_texture, final_sparsity*100);

figure('Name', '最优 L0 分离结果展示 (VM-ANALP)', 'Position', [100, 200, 1600, 400]);
subplot(1, 4, 1); imshow(y, []); title('原始图像 (y)');
subplot(1, 4, 2); imshow(best_y_C, []); title(sprintf('最优卡通 (y_C)\nPSNR: %.2f dB', best_psnr_cartoon));
subplot(1, 4, 3); imshow(best_y_T, []); title(sprintf('最优纹理 (y_T)\nPSNR: %.2f dB', best_psnr_texture));
subplot(1, 4, 4); imshow(reconstructed_best, []); title(sprintf('最优重构 (y_T+y_C)\nPSNR: %.2f dB', psnr_best_total));

