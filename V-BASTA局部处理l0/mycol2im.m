function im = mycol2im(patches, sz, n)
summed = col2imstep(patches, sz + 2*(n-1), [n,n], [1,1]);
im = summed(n:end-n+1, n:end-n+1);
% 上面使用的是full
% 下面使用same方式
% function im = mycol2im(patches, sz, n)
% % MYCOL2IM_SAME Aggregates patches back into an image of size 'sz'.
% %   This function is the inverse of myim2col_same. It sums up overlapping
% %   regions.
% 
%     % Calculate the padding amounts used during the 'same' extraction.
%     p_pre = floor((n - 1) / 2);
%     p_post = ceil((n - 1) / 2);
% 
%     % Determine the size of the padded canvas on which to aggregate.
%     % This size corresponds to the size of the image after 'padarray'
%     % in myim2col_same.
%     canvas_size = sz + p_pre + p_post;
% 
%     % Aggregate patches onto the padded canvas. col2imstep sums overlaps.
%     summed_padded = col2imstep(patches, canvas_size, [n, n], [1, 1]);
% 
%     % Crop the central part of the canvas to get the final image of the
%     % original size 'sz'.
%     im = summed_padded(p_pre + 1 : end - p_post, p_pre + 1 : end - p_post);
% 
% end