function [T2_recon] = eCAMP_PGD(SampledKspace, head_mask, brain_mask, TE, ETL, TVWeight, meanT2)
%
% e-CAMP with Projected Gradient Descent (PGD)
% 
% Inputs:
%		SampledKspace -	Turbo Spin Echo k-space.
%		head_mask     -	The mask for the whole head
%		brain_mask    - The mask for the brain (parenchymal part)
%		TE            - The interval of echoes
%		ETL           -	Echo Train Length
%		TVWeight      -	The weight for Total Variation regularization term
%		meanT2        -	Used to adjust the amplitude of bands straddling the k-space center
%
% Output:
%		T2_recon      - Reconstructed T2 map
%
% Horace Zhehong Zhang, Feb 2023

%% Parameters
% Coils are compressed if there are more than 8 coils
[Nx, Ny, NCoil] = size(SampledKspace);
band = Ny/ETL;
if NCoil > 8 
    CoilCompressionFlag = 1;
else
    CoilCompressionFlag = 0; 
end

% reduce the number of unknowns by half
VCCFlag = 1; % either with virtual conjugate coils
RealValueConstraintFlag = 0; % or only keep the real value during iteration

%% Coil Compression
if CoilCompressionFlag == 1
    NCoilNew = NCoil/2;
    ReshapedKspace = reshape(SampledKspace,Nx*Ny,NCoil);
    [~,~,V] = svd(ReshapedKspace,'econ');
    SCCDATA = ReshapedKspace*V;
    SampledKspace = reshape(SCCDATA(:,1:NCoilNew),Nx,Ny,NCoilNew);
    NCoil = NCoilNew;
end

%% K-space and coil sensitivity map

if mod(ETL,2) ~= 0 % ETL is an odd number

    CentralSampledKspace_Aligned = SampledKspace(:,(ETL-1)/2*band+1:(ETL+1)/2*band,:);

else % ETL is an even number

    % This variable accounts for the asymmetry of the amplitude of the straddling two k-space bands
    T2Decay = exp(-TE/meanT2);
    AsymmetryRatio = zeros(1,ETL/2);
    for ETi = 1:ETL/2
        if ETi ~= ETL/2
            upper = SampledKspace(:,(ETi-1)*band+2:ETi*band+1,:);
            lower = SampledKspace(:,(ETL-ETi)*band+1:(ETL-ETi+1)*band,:);
        else % The central two bands need to avoid the ky=0 line
            upper = SampledKspace(:,(ETi-1)*band+2:ETi*band,:);
            lower = SampledKspace(:,(ETL-ETi)*band+2:(ETL-ETi+1)*band,:);
            CentralBandsRatio = norm(upper(:))/norm(lower(:));
        end
        AsymmetryRatio(ETi) = norm(upper(:))/norm(lower(:))*T2Decay^(ETL-ETi*2+1);
    end
    
    CentralSampledKspace_Aligned = cat(2,squeeze(SampledKspace(:,(ETL/2-1)*band+1:ETL/2*band,:))/CentralBandsRatio*mean(AsymmetryRatio),...
        squeeze(SampledKspace(:,ETL/2*band+1:(ETL/2+1)*band,:)));
end

% Creating conjugate flipped data for VCC
if VCCFlag == 1
    SampledKspace = cat(3,SampledKspace,conj(flip(flip(SampledKspace,1),2)));
    CentralSampledKspace_Aligned = cat(3,CentralSampledKspace_Aligned,conj(flip(flip(CentralSampledKspace_Aligned,1),2)));
    NCoil = NCoil*2;
end

% ESPIRiT for coil sensitivity map estimation
if NCoil == 1
    B1_2DxCoil = ones(Nx,Ny);
else
    eigThresh_1 = 0.05;
    eigThresh_2 = 0.95;
    ksize = [6,6];
    [k,S] = dat2Kernel(CentralSampledKspace_Aligned,ksize);
    idx = max(find(S >= S(1)*eigThresh_1));
    [M,W] = kernelEig(k(:,:,:,1:idx),[Nx,Ny]);
    B1_2DxCoil(:,:,1,:) = M(:,:,:,end).*repmat(W(:,:,end)>eigThresh_2,[1,1,NCoil]);
    B1_2DxCoil = B1_2DxCoil.*head_mask;
end

% get T2w
T2w = sum(ifft2c(SampledKspace).*conj(squeeze(B1_2DxCoil)),3)./sum(squeeze(B1_2DxCoil).*conj(squeeze(B1_2DxCoil)),3);
T2w(isnan(T2w)) = 0;

% T2w image phase incorporated in the coil sensitivity maps
IntPhase = angle(T2w);
B1_2DxCoil = B1_2DxCoil.*exp(1i*IntPhase);

%% e-CAMP
% Band-expanding strategy
switch ETL
    case 4
        ExpansionGroup = [23,13,14];
        k_upl_outerGroup = [40,20,15];
    case 8
        ExpansionGroup = [45,35,36,26,27,17,18];
        k_upl_outerGroup = [30,20,15,10,10,10,5];
    case 9
        ExpansionGroup = [5,56,46,47,37,38,28,29,19];
        k_upl_outerGroup = [1,20,20,15,5,5,5,1,1];
    case 17
        ExpansionGroup = [9,910,810,811,711,712,612,613,513,514,414,415,315,316,216,217,117];
        k_upl_outerGroup = [1,20,20,15,10,5,1,1,1,1,1,1,1,1,1,1,1]; 
    case 19
        ExpansionGroup = [10,1011,911,912,812,813,713,714,614,615,515,516,416,417,317,318,218,219,119];
        k_upl_outerGroup = [1,20,30,30,15,5,1,1,1,1,1,1,1,1,1,1,1,1,1]; 
end
    
for ExpansionIdx = 1:size(ExpansionGroup,2)
    
    %%%%%%%%%%%%%%% Bands extraction %%%%%%%%%%%%%%%%
    if ETL < 10

        stridx = num2str(ExpansionGroup(ExpansionIdx));
        if length(stridx) == 1
            disp(stridx(1))
        else
            disp([stridx(1),'-',stridx(end)])
        end
        ETNo = str2double(stridx(1)):str2double(stridx(end));

        if ExpansionIdx > 1
            stridxLast = num2str(ExpansionGroup(ExpansionIdx-1));
            ETNoLast = str2double(stridxLast(1)):str2double(stridxLast(end));
        end

    else % with more than 10 bands, the denotation is a bit complicated

        stridx = num2str(ExpansionGroup(ExpansionIdx));
        if length(stridx) <= 2 && ExpansionGroup(ExpansionIdx) < ETL
            disp(stridx(1:end))
            ETNo = str2double(stridx(1:end));
        elseif length(stridx) <= 2 && ExpansionGroup(ExpansionIdx) >= ETL
            disp([stridx(1),'-',stridx(end)])
            ETNo = str2double(stridx(1)):str2double(stridx(end));
        else
            disp([stridx(1:end-2),'-',stridx(end-1:end)])
            ETNo = str2double(stridx(1:end-2)):str2double(stridx(end-1:end));
        end

        if ExpansionIdx > 1
            stridxLast = num2str(ExpansionGroup(ExpansionIdx-1));
            if length(stridxLast) <= 2 && ExpansionGroup(ExpansionIdx-1) < ETL
                ETNoLast = str2double(stridxLast(1:end));
            elseif length(stridxLast) <= 2 && ExpansionGroup(ExpansionIdx-1) >= ETL
                ETNoLast = str2double(stridxLast(1)):str2double(stridxLast(end));
            else
                ETNoLast = str2double(stridxLast(1:end-2)):str2double(stridxLast(end-1:end));
            end
        end

    end

    clear data
    if VCCFlag == 1
        for ETi = 1:length(ETNo)
            data(:,band*(ETi-1)+1:band*ETi,:) = SampledKspace(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),1:end/2);
        end
        data = cat(3,data,conj(flip(flip(data,1),2)));
    else
        for ETi = 1:length(ETNo)
            data(:,band*(ETi-1)+1:band*ETi,:) = SampledKspace(:,band*(ETNo(ETi)-1)+1:band*ETNo(ETi),:);
        end
    end
    data = data(:);
    
    %%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%
    % res: concatenated image series and exp(-TE/T2)
    
    if ExpansionIdx == 1 % plain initialization
        res = ones(Nx,Ny,length(ETNo)+1);
    elseif ETNo(1) == ETNoLast(1) % a band appended to the last index
        res = cat(3,res(:,:,1:end-1),res(:,:,end-1).*res(:,:,end),res(:,:,end));
    else % a band appended to the first index
        % alpha shouldn't be 0, especially here
        alpha = res(:,:,end);
        alpha(alpha==0) = 0.01;
        res(:,:,end) = alpha;
        mfirst = res(:,:,1)./(res(:,:,end)+eps);
        % some pixels outside the parenchymal brain are extremely bright
        mfirst(abs(mfirst)>max(abs(mfirst(brain_mask==1)))) = max(abs(mfirst(brain_mask==1))); 
        res = cat(3,mfirst,res(:,:,1:end-1),res(:,:,end));
    end

    %%%%%%%%%%%%%%%%% Projected Gradient Descent %%%%%%%%%%%%%%%%%
    
    k_upl_outer = k_upl_outerGroup(ExpansionIdx);
    res = reshape(ProjGrad(res(:), B1_2DxCoil, data, Nx, Ny, ETNo, band, TVWeight, k_upl_outer, RealValueConstraintFlag, VCCFlag), [Nx Ny length(ETNo)+1]);
    
    %%%%%%%%%%%%%%%%% alpha converted to T2 %%%%%%%%%%%%%%%%%
    
    T2_recon = 1./abs(log(res(:,:,end))/TE); 
    T2_recon(isinf(T2_recon)) = 1;

end