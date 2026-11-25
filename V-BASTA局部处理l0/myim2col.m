function patches = myim2col(im, n)

padded = padarray(im, [n-1,n-1], 'both');
patches = im2colstep(padded, [n,n]);
% 上面被注释掉的是使用了full卷积
% 下面使用same卷积
% function patches = myim2col(im, n)
% % MYIM2COL_SAME Extracts sliding local blocks from an image using 'same' padding.
% %   This ensures the number of extracted patches is equal to the number of
% %   pixels in the original image.
% 
%     % Calculate the required padding for 'same' convolution.
%     % This handles both odd and even filter sizes 'n'.
%     p_pre = floor((n - 1) / 2);
%     p_post = ceil((n - 1) / 2);
%     pad_size = [p_pre, p_post];
% 
%     % Pad the image array. The first argument to padarray is the image,
%     % the second is a vector of padding amounts for [rows, columns].
%     padded = padarray(im, pad_size, 'both');
% 
%     % Extract patches from the padded image with a stride of 1.
%     patches = im2colstep(padded, [n, n], [1, 1]);
% 
% end