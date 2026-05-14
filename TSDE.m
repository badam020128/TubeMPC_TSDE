%% ============================================================
%  OPTIMIZED TSDE TRAINING (TSDE_train.m) - FAST VERSION
%% ============================================================
clear; clc;
load('tsde_training_data.mat'); 

% --- 1. Adatok előkészítése ---
inputs = [X_train; U_train; K_train]; 
targets = R_train; 

% Normalizálás
[in_norm, in_settings] = mapminmax(inputs);
[tar_norm, tar_settings] = mapminmax(targets);

num_ensembles = 5;
ensemble_stage1 = cell(num_ensembles, 1);
ensemble_stage2 = cell(num_ensembles, 1);

% Parallel Pool indítása (ha még nem fut)
if isempty(gcp('nocreate')), parpool; end

% --- 2. STAGE 1: Maradék hiba tanulása (PÁRHUZAMOSAN) ---
fprintf('\n--- Stage 1: Párhuzamos tanítás indítása (5 háló) ---\n');
tic;
parfor i = 1:num_ensembles
    % 'trainscg' - Ez a titok a sebességhez nagy adatnál!
    net = fitnet([32 16], 'trainscg'); 
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 150; % SCG-nél több epoch kellhet, de villámgyors
    
    % Tanítás (GPU esetén: 'useGPU', 'yes')
    ensemble_stage1{i} = train(net, in_norm, tar_norm);
    fprintf('  Stage 1 - %d. háló kész.\n', i);
end
t1 = toc;
fprintf('Stage 1 kész. Idő: %.2f másodperc.\n', t1);

% --- 3. STAGE 2: Bizonytalanság tanulása ---
fprintf('\n--- Stage 2: Bizonytalanság számítása és tanítása ---\n');
% Összesítjük a Stage 1 jóslatokat (ez nem párhuzamosítható könnyen a cell miatt)
stage1_preds = zeros(size(tar_norm));
for i = 1:num_ensembles
    stage1_preds = stage1_preds + ensemble_stage1{i}(in_norm);
end
stage1_preds = stage1_preds / num_ensembles;
stage1_errors = abs(tar_norm - stage1_preds);

tic;
parfor i = 1:num_ensembles
    net_unc = fitnet([16 8], 'trainscg');
    net_unc.trainParam.showWindow = false;
    net_unc.trainParam.epochs = 100;
    
    ensemble_stage2{i} = train(net_unc, in_norm, stage1_errors);
    fprintf('  Stage 2 - %d. háló kész.\n', i);
end
t2 = toc;
fprintf('Stage 2 kész. Idő: %.2f másodperc.\n', t2);

% --- 4. Mentés ---
save('tsde_models.mat', 'ensemble_stage1', 'ensemble_stage2', 'in_settings', 'tar_settings');
disp('Sikeres mentés: tsde_models.mat');