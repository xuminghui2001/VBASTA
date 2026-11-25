function [grad_A, grad_D, g_y] = compute_patch_model_gradients_mask(D_L, D_L_t, A, y_masked, M, image_size, patch_size)
% COMPUTE_PATCH_MODEL_GRADIENTS_MASK 
% 计算图像修复(Inpainting)任务下的梯度和重构误差

if issparse(A)
    A = full(A);
end

% 1: 独立重构所有图像块
reconstructed_patches = D_L * A;

% 2: 叠加合成完整的重构图像 ŷ
y_recon = mycol2im(reconstructed_patches, image_size, patch_size);

% 3: 计算带掩码的重构误差 g(y) (只计算已知像素的误差)
residual = M .* (y_recon - y_masked);
g_y = 0.5 * sum(residual(:).^2);

% 4: 从残差图像中提取出每个位置的残差块
residual_patches_from_image = myim2col(residual, patch_size);

% 5: 计算梯度
grad_A = D_L_t * residual_patches_from_image;
grad_D = residual_patches_from_image * A';

end