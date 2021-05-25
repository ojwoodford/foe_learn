function [params, hmc_step] = foe_learn(X, B, params, niter, hmc_step, step, offset)
%FOE_LEARN  Learn Fields of Experts model parameters
%
%   [params, hs_out] = foe_learn(X, B, params[, niter[, hs_in, [step[, offset]]]])
%
% Learns the parameters of a set of Student t-distributions in a Fields of
% Experts model using Contrastive Divergence. Filters of a size defined by
% B are slid over each training data vector in X.
%
% IN:
%   X - 4-D array containing training data. Each data vector is stored
%       in the first 3 dimensions.
%   B - 3x1 array of filter size along each dimension of the data vectors.
%       The filters will slide (size(X,i)-B(i)+1) times along each
%       dimension.
%   params - (prod(B)+1+offset)xF array of initial model parameters or a
%            scalar, F, indicating how many filters are to be learned (the
%            parameters are initialised randomly).
%   niter - Number of Contrastive Divergence iterations (default: 2000).
%   hs_in - The starting HMC step size (default: 0.01). This value is
%           altered to give an acceptance rate of around 90%.
%   step - The multiplicative factor applied to the gradient before it is
%          added the current parameter estimate (default: 0.01).
%   offset - logical indicating whether the filters require an offset to be
%            added (default: true).
%
% OUT:
%   params - (prod(B)+1+offset)xF array of learned model parameters.
%   hs_out - The final HMC step size.

% Written by O. J. Woodford
% Robotics Research Group
% Department of Engineering Science
% University of Oxford
% ojw@robots.ox.ac.uk

% This software is an implementation of the FoE learning method described
% in:
%   "Fields of Experts for Image-based Rendering"
%   O. J. Woodford, I. D. Reid, P. H. S. Torr and A. W. Fitzgibbon
%   BMVC 2006
% which is, in turn, based on the method described in:
%   "Fields of Experts"
%   Stefan Roth and Michael J. Black
%   CVPR 2005
% If you use this software for publishable work please reference the above
% papers.

if nargin < 7
    offset = true;
    if nargin < 6
        step = 0.01;
        if nargin < 5
            hmc_step = 0.01;
            if nargin < 4
                niter = 2000;
            end
        end
    end
end
offset = double(offset ~= 0);
proposal = @(n) cast(randn([size(X(:,:,:,1)) n]), class(params));

% Select a step size choosing algorithm
%step = @(s, iter) step;
step = @(s, iter) step * (1 / s);
%step = @(s, iter) (1 + 4 * double(iter < 1000)) .* step;
%step = @(s, iter) step * (log(1+max(1000*(grad_thresh*1.1-s), 0)) / s);

% Check inputs
if numel(B) ~= 3
    error('B must contain 3 values');
end
if ndims(X) ~= 4
    error('X must have 4 dimensions');
end
sX = size(X);
if any(B > sX(1:3))
    error('A value in B is too large');
end

flen = prod(B(:)) + 1 + offset;
if numel(params) == 1
    if params > (flen - 1)
        % Randomly initialize
        params = cast(randn(flen-1, params), class(params));
        if offset
            % Set offsets to zero
            params(1,:) = 0;
        end
    else
        % Initialize to the 'nullest' space of the data vectors
        U = double(reshape(X(1:B(1),1:B(2),:,:), [], sX(4))');
        [U, S, V] = svd(U, 0);
        params = cast(V(:,end-params+1:end), class(params));
        clear U S V
        if offset
            params = [zeros(1, size(params, 2), class(params)); params];
        end
    end
    % Set alphas to 1
    params(end+1,:) = 1;
else
    if size(params, 1) ~= flen
        error('params of unexpected size');
    end
    if offset
        params = params([end-1 1:end-2 end],:);
    end
end

% Check the mex file is compiled with the correct parameters
check_mexed_sizes([sX(1:3) B(1:2) offset size(params, 2) 30]);

% Ramp up the number of patches used per iteration from 10%
N = exp((0:niter-1)/(0.3*(niter-1)));
N = N - N(1);
L = min(size(X, 4), 1000);
N = round(L + (size(X, 4) - L) * N / N(end));

for iter = 1:niter
    % Calculate the gradient of our energy function
    L = randperm(size(X, 4));
    O = false(size(L));
    O(L(1:N(iter))) = true;
    [grad, rr] = foe_cd_grad_hmc(params, X(:,:,:,O), proposal(N(iter)), hmc_step);
    grad = grad / N(iter);
    rr = rr / N(iter);
    
    % Update our HMC step size to keep rejections rate constant
    if rr == 0
        hmc_step = hmc_step * 2;
    else
        hmc_step = hmc_step * (1.5 ^ (0.1 - rr));
    end
        
    % Update our parameters (if the rejection rate is acceptable)
    if rr > 0.05 && rr < 0.5
        grad_size = max(abs(grad(:)));
        params(end,:) = log(params(end,:));
        params = params + grad .* step(grad_size, iter);
        params(end,:) = exp(params(end,:));
    end
end
if offset
    params = params([2:end-1 1 end],:);
end
return

function check_mexed_sizes(A)
format = ['#define BLOCK_HEIGHT  %d\r\n' ...
          '#define BLOCK_WIDTH   %d\r\n' ...
          '#define BLOCK_DEPTH   %d\r\n' ...
          '#define FILTER_HEIGHT %d\r\n' ...
          '#define FILTER_WIDTH  %d\r\n' ...
          '#define FILTER_OFFSET %d\r\n' ...
          '#define NFILTS        %d\r\n' ...
          '#define NSTEPS        %d'];
cur_dir = cd;
cd(fileparts(mfilename('fullpath')));
try
    fid = fopen('foe_cd_grad_hmc.h', 'rt');
    B = textscan(fid, format);
    fclose(fid);
    compile = any(A - double(cat(2, B{:})));
catch
    compile = true;
end
if compile
    % Recompile the mex file with the new parameters
    fid = fopen('foe_cd_grad_hmc.h', 'w');
    fprintf(fid, format, A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8));
    fclose(fid);
    try
        mex foe_cd_grad_hmc.cxx
    catch me
        delete('foe_cd_grad_hmc.h');
        cd(cur_dir);
        error(getReport(me));
    end
end
cd(cur_dir);
return
    