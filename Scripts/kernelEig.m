function [EigenVecs, EigenVals] = kernelEig(kernel, imSize)
nc = size(kernel,3);
nv = size(kernel,4);
kSize = [size(kernel,1), size(kernel,2)];

% "rotate kernel to order by maximum variance"
k = permute(kernel,[1,2,4,3]);, k =reshape(k,prod([kSize,nv]),nc);

if size(k,1) < size(k,2)
    [u,s,v] = svd(k);
else
    
    [u,s,v] = svd(k,'econ');
end

k = k*v;
kernel = reshape(k,[kSize,nv,nc]); kernel = permute(kernel,[1,2,4,3]);


KERNEL = zeros(imSize(1), imSize(2), size(kernel,3), size(kernel,4));
for n=1:size(kernel,4)
    KERNEL(:,:,:,n) = (fft2c(zpad(conj(kernel(end:-1:1,end:-1:1,:,n))*sqrt(imSize(1)*imSize(2)), ...
        [imSize(1), imSize(2), size(kernel,3)])));
end
KERNEL = KERNEL/sqrt(prod(kSize));


EigenVecs = zeros(imSize(1), imSize(2), nc, min(nc,nv));
EigenVals = zeros(imSize(1), imSize(2), min(nc,nv));

for n=1:prod(imSize)
    [x,y] = ind2sub([imSize(1),imSize(2)],n);
    mtx = squeeze(KERNEL(x,y,:,:));
    
    %[C,D] = eig(mtx*mtx');
    [C,D,V] = svd(mtx,'econ');
    
    ph = repmat(exp(-i*angle(C(1,:))),[size(C,1),1]);
    C = v*(C.*ph);
    D = real(diag(D));
    EigenVals(x,y,:) = D(end:-1:1);
    EigenVecs(x,y,:,:) = C(:,end:-1:1);
end

end
