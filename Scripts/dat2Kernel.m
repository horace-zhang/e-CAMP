function [kernel, S] = dat2Kernel(data, kSize)
[sx,sy,nc] = size(data);
imSize = [sx,sy] ;

tmp = im2row(data,kSize); [tsx,tsy,tsz] = size(tmp);
A = reshape(tmp,tsx,tsy*tsz);

[U,S,V] = svd(A,'econ');

kernel = reshape(V,kSize(1),kSize(2),nc,size(V,2));
S = diag(S);S = S(:);
end