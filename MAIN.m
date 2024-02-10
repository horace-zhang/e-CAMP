% main function of e-CAMP with PGD and VCC: 
%
% Horace Zhehong Zhang, Feb 2023

clear
addpath(genpath('Scripts'));
warning('off')
tic;

%% Parameters    
DataDirectoryName = 'ExampleData';
DataName = 'meas_MID00059_FID75536_wip_tse_ves_1Image_tse_ves_esp20_res128.mat';
ETL = 9;
meanT2 = []; % [s]
TE = 0.02; % [s]
TVWeight = 2.5;

load([DataDirectoryName,'/',DataName])
load([DataDirectoryName,'/head_mask.mat'])
load([DataDirectoryName,'/brain_mask.mat'])
SampledKspace = SampledKspace*1e8;

%% e-CAMP
T2_recon = eCAMP_PGD(SampledKspace, head_mask, brain_mask, TE, ETL, TVWeight, meanT2);
toc;

%% Display
figure;
subplot(121),imshow(flip(abs(T2_recon.*brain_mask)',1)*1000,[0,180]);colormap default;title('T_2 reconstrucion');colorbar;
hcb = colorbar;hcb.Title.String = "T_2/ms";
load([DataDirectoryName,'/T2_GT.mat'])
subplot(122),imshow(flip(abs(T2_GT.*brain_mask)',1)*1000,[0,180]);colormap default;title('T_2 ground truth');colorbar;
hcb = colorbar;hcb.Title.String = "T_2/ms";
