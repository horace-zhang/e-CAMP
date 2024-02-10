function xoutput = mproject(x0, Nx, Ny, ETNo)
%
% Data projected to the manifold where the image series have T2 decay according to alpha
%
% Horace Zhehong Zhang, Dec 2022

x = x0;

AdamOptmFlag = 1; % Adam optimizer is used now

if AdamOptmFlag == 1
    
    % [xoutput_itm,~,~,~] = fmin_adam_inner(@(x)mprojectObjGrad(x, x0, Nx, Ny, ETNo), x);
    [xoutput_itm,~,~,~] = fmin_adam_inner(@(x)mprojectObjGrad(abs(x), abs(x0), Nx, Ny, ETNo), x);
   
    xoutput = UpdateAllImages(xoutput_itm, Nx, Ny, ETNo);
    
else
    
    maxlsiter = 100;
    gradToll = 1e-30;
    k = 0;
    beta = 0.6;
    t0 = 1;

    g0 = wGradient(x, x0, Nx, Ny, ETNo);

    while(1)

        % backtracking line-search
        f0 = objective(x, x0, Nx, Ny, ETNo);
        t = t0;
        xnew = x - t*g0;
        f1 = objective(xnew, x0, Nx, Ny, ETNo);

        lsiter = 0;

        while f1 > f0 - 0.01*t*abs(g0(:)'*g0(:)) && lsiter < maxlsiter % dx = -t*g0

            lsiter = lsiter + 1;
            t = t * beta;
            xnew = x - t*g0;
            f1  =  objective(xnew, x0, Nx, Ny, ETNo);

        end
        
        if lsiter == maxlsiter
            disp('Reached max line search. Exiting.');
            return;
        end

        % control the number of line searches by adapting the initial step search
        if lsiter > 2
            t0 = t0 * beta;
        end
        if lsiter < 1
            t0 = t0 / beta;
        end
        dx = xnew - x;
        k  = k + 1;
        if k > 5 || norm(dx) < gradToll
            break;
        end
        
        x = xnew;
        g0 = wGradient(x, x0, Nx, Ny, ETNo);
        
    end
    
    xoutput = UpdateAllImages(x, Nx, Ny, ETNo);
end
return

function [obj,grad] = mprojectObjGrad(x, x0, Nx, Ny, ETNo)
obj = objective(x, x0, Nx, Ny, ETNo);
grad = wGradient(x, x0, Nx, Ny, ETNo);


function res = objective(x, x0, Nx, Ny, ETNo)

res = 0;
alpha = x(length(ETNo)*Nx*Ny+1:(length(ETNo)+1)*Nx*Ny);

m1baseFlag = 1;

%%%%%%%%% objective wrt to the image series %%%%%%%%%
if m1baseFlag == 1
    m1 = x(1:Nx*Ny);
    for i = 1:length(ETNo)
        % (alpha^(i-1)*m1_new - mi_original)^2
        diff_1vec = (alpha.^(i-1).*m1 - x0((i-1)*Nx*Ny+1:i*Nx*Ny));
        res = res + diff_1vec(:)'*diff_1vec(:);
    end
else
    % based on the assumption 1. odd ETL; 2. larger TE included first during expansion.
    c = ceil(length(ETNo)/2);
    mc = x((c-1)*Nx*Ny+1:c*Nx*Ny);
    for i = 1:length(ETNo)
        % (alpha^(i-c)*mc_new - mi_original)^2
        diff_1vec = (alpha.^(i-c).*mc - x0((i-1)*Nx*Ny+1:i*Nx*Ny));
        diff_1vec(isnan(diff_1vec)) = 0;
        diff_1vec(isinf(diff_1vec)) = 0;
        res = res + diff_1vec(:)'*diff_1vec(:);
    end
end

%%%%%%%%% objective wrt to the initial alpha %%%%%%%%%
% (alpha_new-alpha_original)^2
diff_1vec = (x(length(ETNo)*Nx*Ny+1:(length(ETNo)+1)*Nx*Ny) - x0(length(ETNo)*Nx*Ny+1:(length(ETNo)+1)*Nx*Ny));
res = res + diff_1vec(:)'*diff_1vec(:);

function grad = wGradient(x, x0, Nx, Ny, ETNo)

alpha = x(length(ETNo)*Nx*Ny+1:(length(ETNo)+1)*Nx*Ny);
grad = x*0;

m1baseFlag = 1;
if m1baseFlag == 1
    %%%%%%%%%%%%%%%%%%% grad wrt to m1 %%%%%%%%%%%%%%%%%%%
    m1 = x(1:Nx*Ny);
    for i = 1:length(ETNo)
        % grad wrt to m1: 2*alpha^(i-1)*(alpha^(i-1)*m1_new - mi_original)
        grad(1:Nx*Ny) = grad(1:Nx*Ny) + 2*alpha.^(i-1).*(alpha.^(i-1).*m1 - x0((i-1)*Nx*Ny+1:i*Nx*Ny));
    end
else
    %%%%%%%%%%%%%%%%%%% grad wrt to mc %%%%%%%%%%%%%%%%%%%
    % based on the assumption 1. odd ETL; 2. larger TE included first during expansion.
    c = ceil(length(ETNo)/2);
    mc = x((c-1)*Nx*Ny+1:c*Nx*Ny);
    for i = 1:length(ETNo)
        % grad wrt to mc: 2*alpha^(i-c)*(alpha^(i-c)*mc_new - mi_original)
        grad((c-1)*Nx*Ny+1:c*Nx*Ny) = grad((c-1)*Nx*Ny+1:c*Nx*Ny) + 2*alpha.^(i-c).*(alpha.^(i-c).*mc - x0((i-1)*Nx*Ny+1:i*Nx*Ny));
    end
    gradtemp = grad((c-1)*Nx*Ny+1:c*Nx*Ny);
    gradtemp(isnan(gradtemp)) = 0;
    gradtemp(isinf(gradtemp)) = 0;
    grad((c-1)*Nx*Ny+1:c*Nx*Ny) = gradtemp;
end

%%%%%%%%%%%%%%%%% grad wrt to alpha %%%%%%%%%%%%%%%%%
Sidx = length(ETNo)*Nx*Ny+1;
Eidx = (length(ETNo)+1)*Nx*Ny;
for i = 2:length(ETNo)
    if m1baseFlag == 1
        % grad wrt to alpha (the first term): 2*(i-1)*m1*alpha^(i-2)*(alpha^(i-1)*m1_new - mi_original)
        grad(Sidx:Eidx) = grad(Sidx:Eidx) + 2*(i-1)*m1.*alpha.^(i-2).*(alpha.^(i-1).*m1 - x0((i-1)*Nx*Ny+1:i*Nx*Ny));
    else
        % grad wrt to alpha (the first term): 2*(i-c)*mc*alpha^(i-c-1)*(alpha^(i-c)*mc_new - mi_original)
        grad(Sidx:Eidx) = grad(Sidx:Eidx) + 2*(i-c)*mc.*alpha.^(i-c-1).*(alpha.^(i-c).*mc - x0((i-1)*Nx*Ny+1:i*Nx*Ny));
        gradtemp = grad(Sidx:Eidx);
        gradtemp(isnan(gradtemp)) = 0;
        gradtemp(isinf(gradtemp)) = 0;
        grad(Sidx:Eidx) = gradtemp;
    end
end

% grad wrt to alpha (the last term): 2*(alpha_new-alpha_original)
grad(Sidx:Eidx) = grad(Sidx:Eidx) + 2*(x(Sidx:Eidx) - x0(Sidx:Eidx));

function x = UpdateAllImages(x, Nx, Ny, ETNo)

%%%%%%%%%% constraint of m & alpha %%%%%%%%%%

m1baseFlag = 1;

if m1baseFlag == 1
    % m1 constraint
    m1 = x(1:Nx*Ny);
    m1(m1<0) = 0;
    x(1:Nx*Ny) = m1;
else
    % based on the assumption 1. odd ETL; 2. larger TE included first during expansion.
    c = ceil(length(ETNo)/2);
    % mc constraint
    mc = x((c-1)*Nx*Ny+1:c*Nx*Ny);
    mc(mc<0) = 0;
    x((c-1)*Nx*Ny+1:c*Nx*Ny) = mc;
end

% alpha constraint
alpha = x(length(ETNo)*Nx*Ny+1:(length(ETNo)+1)*Nx*Ny);
alpha(alpha<0) = 0;
alpha(abs(alpha)>0.9999) = 0.9999; % T2 is not supposed to be too large
x(length(ETNo)*Nx*Ny+1:(length(ETNo)+1)*Nx*Ny) = alpha;

%%%%%%%% alignment of the series of m %%%%%%%%
if m1baseFlag == 1
    for i = 1:length(ETNo)
        x((i-1)*Nx*Ny+1:i*Nx*Ny) = m1.*alpha.^(i-1);
    end
else
    for i = 1:length(ETNo)
        x((i-1)*Nx*Ny+1:i*Nx*Ny) = mc.*alpha.^(i-c);
    end
    x(isnan(x)) = 0;
    x(isinf(x)) = 0;
end