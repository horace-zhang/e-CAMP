function x = ProjGrad(x0, B1_2DxCoil, data, Nx, Ny, ETNo, band, TVWeight, k_upl_outer, RealValueConstraintFlag, VCCFlag)
%
% Simultaneous optimization of the image series and alpha with Projected Gradient Descent
%
% Horace Zhehong Zhang, Dec 2022

x = x0;
    
gradToll = 1e-30;
k = 0;
beta = 0.6;
maxlsiter = 100;
t0 = 0.97;

g0 = wGradient(x, B1_2DxCoil, data, Nx, Ny, ETNo, band, TVWeight, RealValueConstraintFlag, VCCFlag);

vff1 = zeros(k_upl_outer,1);

while(1)

    k = k + 1;
    if k > k_upl_outer
        break;
    end

    f0 = objective(x, data, B1_2DxCoil, Nx, Ny, ETNo, band, TVWeight, VCCFlag);

    t = t0;
    xnew = x - t * g0;
    dx = -t * g0;
    f1 = objective(xnew, data, B1_2DxCoil, Nx, Ny, ETNo, band, TVWeight, VCCFlag);

    lsiter = 0;

    while f1 > f0 - 0.01*abs(g0(:)'*dx(:)) && lsiter < maxlsiter
        lsiter = lsiter + 1;
        t = t * beta;
        xnew = x - t * g0;
        dx = -t * g0;
        f1 = objective(xnew, data, B1_2DxCoil, Nx, Ny, ETNo, band, TVWeight, VCCFlag);
    end

    if lsiter == maxlsiter
        disp('Reached max line search. Exiting.');
        return;
    end

    if lsiter > 2
        t0 = t0 * beta;
    end
    if lsiter < 1
        t0 = t0 / beta;
    end
    if norm(dx) < gradToll
        break;
    end
    x = xnew;

    if length(ETNo) > 1
        x = mproject(x, Nx, Ny, ETNo);
    end

    vff1(k) = f1;

    g0 = wGradient(x, B1_2DxCoil, data, Nx, Ny, ETNo, band, TVWeight, RealValueConstraintFlag, VCCFlag);

end

return


function res = objective(x, data, B1_2DxCoil, Nx, Ny, ETNo, band, TVWeight, VCCFlag)

ImageSeriesx = reshape(x,[Nx,Ny,length(ETNo)+1]);
ImageSeriesx = ImageSeriesx(:,:,1:end-1);
ImageSeriesxCoil = ImageSeriesx.*B1_2DxCoil;
FullKspace = fft2c(ImageSeriesxCoil);

Atx = squeeze(FullKspace(:,1:band*length(ETNo),1,:))*0;

if VCCFlag == 0
    for ETi = 1:length(ETNo)
        Atx(:,band*(ETi-1)+1:band*ETi,:) = FullKspace(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),ETi,:);
    end
else
    for ETi = 1:length(ETNo)
        Atx(:,band*(ETi-1)+1:band*ETi,1:end/2) = squeeze(FullKspace(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),ETi,1:end/2));
        Atx(:,band*(ETi-1)+1:band*ETi,end/2+1:end) = squeeze(FullKspace(:,Ny-band*ETNo(end+1-ETi)+1:Ny-band*(ETNo(end+1-ETi)-1),end+1-ETi,end/2+1:end));
    end
end
Atx = Atx(:);
diff = Atx - data;
J1 = diff(:)'*diff(:);

if TVWeight > 0
    TVop = TVOP;
    DXFMtx = cat(3,ImageSeriesx,ImageSeriesx);
    for ETi = 1:length(ETNo)
        DXFMtx(:,:,2*ETi-1:2*ETi) = TVop*ImageSeriesx(:,:,ETi);
    end
    DXFMtx = DXFMtx(:);
    J2 = sum((DXFMtx.*conj(DXFMtx)+1e-15).^0.5);
else
    J2 = 0;
end

res = J1 + TVWeight*J2;

function grad = wGradient(x, B1_2DxCoil, data, Nx, Ny, ETNo, band, TVWeight, RealValueConstraintFlag, VCCFlag)

gradObj = gJ1x(x, B1_2DxCoil, data, Nx, Ny, ETNo, band, RealValueConstraintFlag, VCCFlag);

if TVWeight > 0
    gradTV = gTV(x, Nx, Ny, ETNo);
else
    gradTV = 0;
end

grad = gradObj + TVWeight.*gradTV;

function grad = gJ1x(x, B1_2DxCoil, data, Nx, Ny, ETNo, band, RealValueConstraintFlag, VCCFlag)

Ncoil = size(B1_2DxCoil,4);
datare = reshape(data,[Nx,length(ETNo)*band,Ncoil]);
ImageSeriesx = reshape(x,[Nx,Ny,length(ETNo)+1]);
% extracting the image part only
ImageSeriesx = ImageSeriesx(:,:,1:end-1);
ImageSeriesxCoil = ImageSeriesx.*B1_2DxCoil;

FullKspace = fft2c(ImageSeriesxCoil);

Atx_b = FullKspace*0;

if VCCFlag == 0
    for ETi = 1:length(ETNo)
        Atx_b(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),ETi,:) = squeeze(FullKspace(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),ETi,:)) - datare(:,band*(ETi-1)+1:band*ETi,:);
    end
else
    for ETi = 1:length(ETNo)
        Atx_b(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),ETi,1:end/2) = squeeze(FullKspace(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),ETi,1:end/2)) - datare(:,band*(ETi-1)+1:band*ETi,1:end/2);
        Atx_b(:,Ny-band*ETNo(end+1-ETi)+1:Ny-band*(ETNo(end+1-ETi)-1),end+1-ETi,end/2+1:end) = ...
            squeeze(FullKspace(:,Ny-band*ETNo(end+1-ETi)+1:Ny-band*(ETNo(end+1-ETi)-1),end+1-ETi,end/2+1:end)) - datare(:,band*(ETi-1)+1:band*ETi,end/2+1:end);
    end
end

DiffImageSeriesxCoil = ifft2c(Atx_b);

DiffImageSeriesx = sum(DiffImageSeriesxCoil.*conj(B1_2DxCoil),4);
Temp_gradObj = 2*DiffImageSeriesx;

if RealValueConstraintFlag == 1
    grad = real(Temp_gradObj);
else
    grad = Temp_gradObj;
end

% concatenating the grad wrt image with the grad wrt alpha (0)
grad = cat(3,grad,grad(:,:,1)*0);
grad = grad(:);


function grad = gTV(x, Nx, Ny, ETNo)

x = reshape(x,[Nx Ny length(ETNo)+1]);
% extracting the image part only
x = x(:,:,1:end-1);

TVop = TVOP;

Dx = x;
for i = 1:length(ETNo)
    Dx(:,:,2*i-1:2*i) = TVop*x(:,:,i);
end
G = Dx.*(Dx.*conj(Dx) + 1e-15).^(-1/2);

grad = x;
for i=1:length(ETNo)
    grad(:,:,i)=TVop'*G(:,:,2*i-1:2*i);
end
% concatenating the grad wrt image with the grad wrt alpha (0)
grad = cat(3,grad,grad(:,:,1)*0);

grad = grad(:);
