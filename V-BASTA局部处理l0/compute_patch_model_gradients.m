function [grad_A, grad_D, g_y] = compute_patch_model_gradients(D_L, D_L_t, A, y, image_size, patch_size)
% COMPUTE_PATCH_MODEL_GRADIENTS_NEW (严格定义版)
% 计算基于图像块叠加模型的目标函数 H_conv 的梯度和重构误差。

% 确保输入为非稀疏格式以用于后续计算
if issparse(A)
    A = full(A);
end

% 步骤 1: 独立重构所有图像块
reconstructed_patches = D_L * A;

% 步骤 2: 叠加合成完整的重构图像 ŷ
y_recon = mycol2im(reconstructed_patches, image_size, patch_size);

% 步骤 3: 计算重构误差 g(y)
residual = y_recon - y;
g_y = 0.5 * sum(residual(:).^2);

% 步骤 4: 从残差图像中提取出每个位置的残差块
residual_patches_from_image = myim2col(residual, patch_size);

% 步骤 5: 计算梯度
% 5.1 计算对稀疏系数 A 的梯度
grad_A = D_L_t * residual_patches_from_image;

% 5.2 计算对字典 D_L 的梯度
grad_D = residual_patches_from_image * A';

end