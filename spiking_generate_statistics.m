%% compute statistics for constrained RNNs: SPIKING
% written by danyal akarca, imperial college london & university of cambridge, 2024

% this code allows for computation of relevant statistics for the tested spiking RNNs (SHD task)

% set up code
clear; clc;
repo = '/Users/da04/Desktop/repo'; % set repo directory
cd(repo); % cd to repo
load('data/spiking_data.mat'); % load spiking data
savedir = fullfile(repo,'data'); % save directory when running the below code
addpath('prereq/2019_03_03_BCT/'); % load prereqisites

% for details on the trained rate networks, see:
% Achterberg, J. % Akarca, D., et al. 
% Spatially embedded recurrent neural networks reveal widespread 
% links between structural and functional neuroscience findings. 
% Nature Machine Intelligence 5, 1369–1381 (2023). 
% https://doi.org/10.1038/s42256-023-00748-9

%% compute relevant statistics on spiking networks (~5 minutes runtime)

% initialise networks
N = 100; % number of neurons
ntype = 2; % number of network setups
nepoch = 50; % number of epochs
nnet = [100,100]; % number of networks in each group
nstat = 6; % number of computed statistics
spiking_rnns_entropy_statistics = struct; % initialise statistics for network types
spiking_rnns_entropy_statistics.L1 = zeros(101,50,nstat);
spiking_rnns_entropy_statistics.seRNN = zeros(101,50,nstat);
stat_label = string({... % set labels
    'Shannon entropy (W)',... % shannon of absolute weight matrix
    'Shannon entropy (C)',... % shannon of normalised communicability matrix of absolute weight matrix
    'Modularity (Q)',... % directed modularity
    '\lambda_{max}',... % leading eigenvalue (non absolute weight matrix)
    'Spectral entropy (H(W_{lambda}))',... % spectral entropy (non absolute weight matrix)
    'Total weight'}); % total absolute weights
rate_rnns.entropy_statistics.statistic_labels = stat_label;
type_label = string({'L1','seRNN'}); % types of networks

% loop and compute statistics
for type = 1:ntype; % loop over network type
    for epoch = 1:nepoch; % loop over epochs
        for net = 1:nnet(type) % loop over networks
                % get network
                if type == 1;
                    w = squeeze(spiking_data.L1.hidden(net,epoch,:,:));
                else if type == 2;
                        w = squeeze(spiking_data.seRNN.hidden(net,epoch,:,:));
                    end
                end
                % absolute the weight matrix
                aabs = abs(w);
                % 1. shannon entropy on absolute matrix
                p = zeros(N);
                for i = 1:N 
                    for j = 1:N 
                        p(i,j) = (aabs(i, j))/(sum(aabs(i, :))); % build the probability distribution
                    end 
                end
                h = zeros(N,1);
                for i = 1:N;
                    h(i) = -sum(p(i,:).*log2(p(i,:)));
                end
                se_aabs = mean(h); 
                % 2. shannon entropy on absolute normalised communicability matrix
                p = zeros(N);
                s = diag(sum(aabs,2));
                adj = (s^-.5)*aabs*(s^-.5);
                ncabs = expm(adj);
                for i = 1:N 
                    for j = 1:N 
                        p(i,j) = (ncabs(i,j))/(sum(ncabs(i,:))); 
                    end 
                end
                h = zeros(N,1);
                for i = 1:N;
                    h(i) = -sum(p(i,:).*log2(p(i,:)));
                end
                se_ncabs = mean(h);
                % 3. modularity directed of the absolute weighted network
                [~,qd_aabs] = modularity_dir(aabs);
                % 4. leading eigenvalue of the weight matrix
                ew = eig(w);
                [evw,f] = eig(w);
                new = abs(ew); % abs computes the norm
                pw = new./sum(new);
                ksw = log(max(real(ew)));
                % 5. spectral entropy of weight matrix eigenvalues (norm)
                sew = -sum(pw.*log2(pw),'omitnan');
                % 6. total weight
                tw = sum(aabs,'all');
                % keep the statistics for each network type
                if type == 1; % L1 networks
                    spiking_rnns_entropy_statistics.L1(net,epoch,1) = se_aabs; % shannon entropy on absolute weight matrix
                    spiking_rnns_entropy_statistics.L1(net,epoch,2) = se_ncabs; % shannon entropy on absolute normalised communicability matrix
                    spiking_rnns_entropy_statistics.L1(net,epoch,3) = qd_aabs; % directed q statistic on absolute weight matrix
                    spiking_rnns_entropy_statistics.L1(net,epoch,4) = ksw; % leading eigenvalue of weight matrix
                    spiking_rnns_entropy_statistics.L1(net,epoch,5) = sew; % spectral entropy of weight matrix eigenvalues
                    spiking_rnns_entropy_statistics.L1(net,epoch,6) = sum(aabs,'all'); % total weight of absolute weight matrix
                end
                if type == 2; % seRNNs
                    spiking_rnns_entropy_statistics.seRNN(net,epoch,1) = se_aabs; 
                    spiking_rnns_entropy_statistics.seRNN(net,epoch,2) = se_ncabs;
                    spiking_rnns_entropy_statistics.seRNN(net,epoch,3) = qd_aabs;
                    spiking_rnns_entropy_statistics.seRNN(net,epoch,4) = ksw;
                    spiking_rnns_entropy_statistics.seRNN(net,epoch,5) = sew;
                    spiking_rnns_entropy_statistics.seRNN(net,epoch,6) = sum(aabs,'all');
                end
                % display
                disp(sprintf('spiking network %g of %g (epoch %g) %s networks complete',...
                    net,nnet(type),epoch,type_label(type)));
        end
    end
end

%% save the data into the save directory

cd(savedir); % change into the save directory
save('spiking_rnns_entropy_statistics.mat','spiking_rnns_entropy_statistics','-v7.3'); % save the struct