function [im_lcn, local_mean, local_std_clipped] = local_contrast_normalise(im)
% LOCAL_CONTRAST_NORMALISE 实现了与 CreateImages.m 中完全一致的 LCN 操作
%
% 输入:
%   im:      输入的灰度图像 (应为 double 或 single 类型)
%
% 输出:
%   im_lcn:             LCN 处理后的图像
%   local_mean:         用于归一化的局部均值图
%   local_std_clipped:  经过中值阈值处理的局部标准差图

    % --- Logic copied directly from CreateImages.m lines 313-366 ---
    
    % 原始脚本中使用的硬编码高斯核
    k = fspecial('gaussian',[13 13],3*1.591);
    k2 = fspecial('gaussian',[13 13],3*1.591);
    
    % 检查核是否相同 (如原始脚本所示)
    if(all(k(:)==k2(:)))
        SAME_KERNELS=1;
    else
        SAME_KERNELS=0;
    end
    
    dim = double(im);
    
    % 使用 rconv2 计算局部均值和二阶矩
    lmn = rconv2(dim,k);
    lmnsq = rconv2(dim.^2,k2);
    
    if(SAME_KERNELS)
        lmn2 = lmn;
    else
        lmn2 = rconv2(dim,k2);
    end
    
    lvar = lmnsq - lmn2.^2;
    lvar(lvar<0) = 0; % 避免数值问题
    lstd = sqrt(lvar);
    
    % 来自原始脚本的复杂中值阈值处理
    q=sort(lstd(:));
    lq = round(length(q)/2);
    th = q(lq);
    if(th==0)
        q_non_zero = nonzeros(q);
        if(~isempty(q_non_zero))
            lq = round(length(q_non_zero)/2);
            th = q_non_zero(lq);
        else
            th = 0;
        end
    end
    
    local_std_clipped = lstd;
    local_std_clipped(local_std_clipped <= th) = th;
    local_std_clipped(local_std_clipped(:)==0) = eps;
    
    % 最终的 LCN 计算
    local_mean = lmn;
    im_lcn = (dim - local_mean) ./ local_std_clipped;
end