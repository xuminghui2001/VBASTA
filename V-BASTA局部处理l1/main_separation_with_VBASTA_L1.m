%% Cartoon-Texture Separation using VM-ANALP Framework (With Best Tracking)
% =========================================================================
% 特性:
% 1. 采用 Variable Metric (可变度量) 加速纹理分量 (D, A) 的学习
% 2. 结合 TV Denoising 提取平滑的卡通分量 y_C
% 3. 继承 VM-ANALP 的智能缓存与 Warm Start 步长机制
% 4. 【新增】独立追踪卡通(y_C)和纹理(y_T)的最高PSNR，并支持智能提前终止
% 5. 【新增】三窗口实时监控：分离图像、学习字典、PSNR收敛曲线
% =========================================================================

close all; clear; clc;
addpath(genpath('.')); 

%% 1. 参数设置
fprintf('--- 1. 参数设置 (VM-ANALP 分离追踪版) ---\n');
m = 100;              
patch_size = 11;      
lambda1 = 0.1;        % 纹理稀疏度权重
zeta = 0.025;         % 卡通平滑度权重 (TV项)

% --- VM-ANALP 优化参数 ---
MAXITER = 1000;       
max_backtrack_iter = 30; 
gamma1 = 0.006;       % S_alpha 初始缩放
gamma2 = 0.006;       % S_D 初始缩放
tao_bt = 1.5;         % 回溯衰减倍率

% --- 可视化与智能停止参数 ---
plot_interval = 5; % 每 N 次迭代更新一次图像
patience = 30;     % 提前终止的耐心值：连续30次无改进则停止

%% 2. 加载与合成数据
fprintf('--- 2. 加载数据 ---\n');
load("datasets\separation\cat4.mat"); % 假设包含 imgc (卡通)
load("datasets\separation\fense2.mat"); % 假设包含 fense2 (纹理)

I_true = im2double(imgc); 
M_true = 0.4 * im2double(imresize(fense2, [256, 256]));
y = I_true + M_true; % 合成观测图像
numpic = 1; 

image_sizes{1} = size(y);
dummy_patches = myim2col(zeros(image_sizes{1}), patch_size);
N_l_vec(1) = size(dummy_patches, 2);

%% 3. 初始化变量与 VM 缓存
fprintf('--- 3. 初始化变量 ---\n');
d = patch_size^2;
D = randn(d, m); D = proxomiga2_2D(D);
A_cells{1} = zeros(m, N_l_vec(1));
y_C = zeros(size(y)); 

% 步长记忆缓存
tau1_history = gamma1 * ones(1, numpic);
tau2_history = gamma2;

% --- 【新增】初始化最优结果追踪变量 ---
best_psnr_cartoon = -Inf; best_y_C = [];
best_psnr_texture = -Inf; best_D_for_texture = []; best_A_for_texture = {};
iterations_without_improvement = 0;

history.psnr_total = zeros(1, MAXITER);
history.psnr_cartoon = zeros(1, MAXITER);
history.psnr_texture = zeros(1, MAXITER);
sumtime = 0;

%% 4a. 实时可视化设置
fprintf('--- 4a. 初始化可视化窗口 ---\n');
hFig1 = figure('Name', 'VM-ANALP 实时分离结果', 'Position', [50, 400, 1600, 400]);
hFig2 = figure('Name', 'VM-ANALP 实时学习字典', 'Position', [50, 50, 600, 600]);
hFig3 = figure('Name', 'VM-ANALP 实时PSNR曲线', 'Position', [700, 50, 600, 400]);

%% 4b. VM-ANALP + TV 交替优化主循环
fprintf('--- 4b. VM-ANALP + TV 交替优化主循环开始 ---\n');
for outerIter = 1:MAXITER
    tic;
    
    % ==========================================================
    % STEP 1: 更新纹理分量 (D, A) - VM-ANALP 逻辑
    % ==========================================================
    y_target = y - y_C; % 纹理的目标是残差
    
    % --- 缓存 S_alpha (可变度量矩阵) ---
    abs_D = abs(D);
    D_sum = sum(abs_D, 2);
    D_rep = repmat(D_sum, 1, N_l_vec(1));
    Img_overlap = mycol2im(D_rep, image_sizes{1}, patch_size);
    P_overlap = myim2col(Img_overlap, patch_size);
    S_alpha = abs_D' * P_overlap + 1e-8; 
    
    % --- 子步 A: 更新系数 A ---
    [grad_A, ~, H_old] = compute_patch_model_gradients(D, D', A_cells{1}, y_target, image_sizes{1}, patch_size);
    tau1_curr = max(gamma1, tau1_history(1) / 1.2);
    
    for k_bt = 1:max_backtrack_iter
        S_effective = tau1_curr * S_alpha;
        v_A = A_cells{1} - grad_A ./ S_effective;
        A_candidate = soft_threshold(v_A, lambda1 ./ S_effective);
        
        % 下降引理检查
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
    
    % --- 子步 B: 更新字典 D ---
    [~, grad_D, H_old_D] = compute_H_gradients_in_parallel(D, A_cells, {y_target}, image_sizes, patch_size);
    
    % 计算 S_D
    abs_A = abs(A_cells{1});
    X_mat = repmat(sum(abs_A, 1), d, 1);
    S_D = myim2col(mycol2im(X_mat, image_sizes{1}, patch_size), patch_size) * abs_A' + 1e-8;
    
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
    
    fprintf('迭代 %d/%d: PSNR(总)=%.2f, PSNR(纹理)=%.2f, PSNR(卡通)=%.2f, 耗时=%.2fs\n', ...
            outerIter, MAXITER, psnr_total, psnr_texture, psnr_cartoon, elapsed_time);
            
    % --- 最优结果追踪 ---
    improved = false;
    if psnr_cartoon > best_psnr_cartoon
        best_psnr_cartoon = psnr_cartoon;
        best_y_C = y_C;
        improved = true;
        fprintf('       *** 找到新的最优卡通结果 (PSNR: %.4f dB) ***\n', best_psnr_cartoon);
    end
    if psnr_texture > best_psnr_texture
        best_psnr_texture = psnr_texture;
        best_D_for_texture = D;
        best_A_for_texture = A_cells;
        improved = true;
        fprintf('       *** 找到新的最优纹理结果 (PSNR: %.4f dB) ***\n', best_psnr_texture);
    end
    
    % --- 智能提前终止 ---
    if improved
        iterations_without_improvement = 0; 
    else
        iterations_without_improvement = iterations_without_improvement + 1; 
    end
    
    if iterations_without_improvement >= patience
        fprintf('\n--- 在连续 %d 次迭代中无改进，触发提前终止。 ---\n', patience);
        history.psnr_total = history.psnr_total(1:outerIter);
        history.psnr_texture = history.psnr_texture(1:outerIter);
        history.psnr_cartoon = history.psnr_cartoon(1:outerIter);
        break; 
    end

    % ==========================================================
    % STEP 4: 实时可视化更新
    % ==========================================================
    if mod(outerIter, plot_interval) == 0 || outerIter == 1 || outerIter == MAXITER
        
        % 图 1：分离结果
        if ~isgraphics(hFig1, 'figure'), hFig1 = figure('Name', 'VM-ANALP 实时分离结果', 'Position', [50, 400, 1600, 400]); end
        figure(hFig1); 
        subplot(1, 4, 1); imshow(y, []); title('原始图像 (y)');
        subplot(1, 4, 2); imshow(y_C, []); title(sprintf('卡通分量 (y_C)\nPSNR: %.2f dB', psnr_cartoon));
        subplot(1, 4, 3); imshow(y_T_reconstructed, []); title(sprintf('纹理分量 (y_T)\nPSNR: %.2f dB', psnr_texture));
        subplot(1, 4, 4); imshow(reconstructed_img, []); title(sprintf('重构图像 (y_T+y_C)\nPSNR: %.2f dB', psnr_total));
        
        % 图 2：字典
        if ~isgraphics(hFig2, 'figure'), hFig2 = figure('Name', 'VM-ANALP 实时学习字典', 'Position', [50, 50, 600, 600]); end
        figure(hFig2); 
        showDictionary(D);
        title(sprintf('VM-ANALP 学习的字典 (迭代 %d)', outerIter));
        
        % 图 3：PSNR 曲线
        if ~isgraphics(hFig3, 'figure'), hFig3 = figure('Name', 'VM-ANALP 实时PSNR曲线', 'Position', [700, 50, 600, 400]); end
        figure(hFig3); 
        plot(1:outerIter, history.psnr_total(1:outerIter), 'b-', 'DisplayName', 'PSNR (总)'); hold on;
        plot(1:outerIter, history.psnr_texture(1:outerIter), 'r--', 'DisplayName', 'PSNR (纹理)');
        plot(1:outerIter, history.psnr_cartoon(1:outerIter), 'g:', 'DisplayName', 'PSNR (卡通)'); hold off;
        xlabel('迭代次数'); ylabel('PSNR (dB)'); title('VM-ANALP 实时PSNR收敛曲线');
        legend show; grid on; xlim([0, MAXITER]);
        
        drawnow;
    end
end
fprintf('\n--- VM-ANALP 分离完成，总耗时: %.2f 秒 ---\n', sumtime);

%% 5. 保存并展示最优分离结果
fprintf('--- 5. 保存并展示追踪到的最优结果 ---\n');

% 重构最优结果
best_y_T = mycol2im(best_D_for_texture * best_A_for_texture{1}, image_sizes{1}, patch_size);
reconstructed_best = best_y_C + best_y_T;
psnr_best_total = psnr(reconstructed_best, y);

% 保存文件
save('best_separation_results_vmanalp.mat', ...
     'best_y_C', 'best_y_T', 'best_D_for_texture', 'best_A_for_texture', ...
     'best_psnr_cartoon', 'best_psnr_texture', 'psnr_best_total');
fprintf('最优结果已保存到 best_separation_results_vmanalp.mat 文件中。\n');
fprintf('最终结果: 最优卡通 PSNR = %.4f dB, 最优纹理 PSNR = %.4f dB\n', best_psnr_cartoon, best_psnr_texture);

% 绘制最终的最优分离图
figure('Name', '最优分离结果展示 (VM-ANALP)', 'Position', [100, 200, 1600, 400]);
subplot(1, 4, 1); imshow(y, []); title('原始图像 (y)');
subplot(1, 4, 2); imshow(best_y_C, []); title(sprintf('最优卡通 (y_C)\nPSNR: %.2f dB', best_psnr_cartoon));
subplot(1, 4, 3); imshow(best_y_T, []); title(sprintf('最优纹理 (y_T)\nPSNR: %.2f dB', best_psnr_texture));
subplot(1, 4, 4); imshow(reconstructed_best, []); title(sprintf('最优重构 (y_T+y_C)\nPSNR: %.2f dB', psnr_best_total));
% 这里的保存文件名和变量名必须与 L0 脚本中的 load() 函数严格对齐！
save('best_separation_results_vmanalp.mat', ...
     'best_y_C', 'best_y_T', 'best_D_for_texture', 'best_A_for_texture', ...
     'best_psnr_cartoon', 'best_psnr_texture', 'psnr_best_total');

fprintf('最优L1结果已成功保存到 best_separation_results_vmanalp.mat 文件中！\n');
fprintf('现在你可以去运行 L0 脚本进行极速热启动了。\n');