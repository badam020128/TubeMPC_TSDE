%% ============================================================
%  OPTIMIZED TSDE TRAINING (TSDE_train.m) - FAST & DEEP VERSION
%% ============================================================
disp('--- Adatok betöltése... ---');
load('tsde_training_data.mat'); 

% --- 1. Adatok előkészítése ---
inputs = [X_train; U_train; K_train]; 
targets = R_train; 

% Normalizálás
[in_norm, in_settings] = mapminmax(inputs);
[tar_norm, tar_settings] = mapminmax(targets);

% Növeljük 5-re az ensemble számot a nagyobb robusztusságért
num_ensembles = 4; 
ensemble_stage1 = cell(num_ensembles, 1);
ensemble_stage2 = cell(num_ensembles, 1);

% Parallel Pool indítása (ha még nem fut)
if isempty(gcp('nocreate')), parpool; end

% --- 2. STAGE 1: Maradék hiba tanulása (PÁRHUZAMOSAN) ---
fprintf('\n--- Stage 1: Párhuzamos tanítás indítása (%d háló) ---\n', num_ensembles);
tic;
parfor i = 1:num_ensembles
    % Növeltük a kapacitást: [64 32] neuron
    net = fitnet([64 32], 'trainscg'); 
    net.trainParam.showWindow = false;
    
    % --- KRITIKUS TANÍTÁSI PARAMÉTEREK ---
    net.trainParam.epochs = 500;    % 150 helyett! Hagyjuk kibontakozni
    net.trainParam.max_fail = 20;    % Ne álljon le, ha átmenetileg ugrál a validációs hiba
    net.trainParam.min_grad = 1e-6;  % Sokkal finomabb hangolást engedünk
    
    % Tanítás
    ensemble_stage1{i} = train(net, in_norm, tar_norm);
    fprintf('  Stage 1 - %d. háló kész.\n', i);
end
t1 = toc;
fprintf('Stage 1 kész. Idő: %.2f másodperc.\n', t1);

% --- 3. STAGE 2: Bizonytalanság tanulása ---
fprintf('\n--- Stage 2: Bizonytalanság számítása és tanítása ---\n');
% Összesítjük a Stage 1 jóslatokat
stage1_preds = zeros(size(tar_norm));
for i = 1:num_ensembles
    stage1_preds = stage1_preds + ensemble_stage1{i}(in_norm);
end
stage1_preds = stage1_preds / num_ensembles;

% Az abszolút hiba, amit a Stage 2-nek meg kell tanulnia
stage1_errors = abs(tar_norm - stage1_preds);

tic;
parfor i = 1:num_ensembles
    % A bizonytalanságot elég egy kisebb hálónak megtanulnia
    net_unc = fitnet([32 16], 'trainscg');
    net_unc.trainParam.showWindow = false;
    
    net_unc.trainParam.epochs = 400; 
    net_unc.trainParam.max_fail = 20;
    
    ensemble_stage2{i} = train(net_unc, in_norm, stage1_errors);
    fprintf('  Stage 2 - %d. háló kész.\n', i);
end
t2 = toc;
fprintf('Stage 2 kész. Idő: %.2f másodperc.\n', t2);

% --- 4. Mentés ---
save('tsde_models.mat', 'ensemble_stage1', 'ensemble_stage2', 'in_settings', 'tar_settings', 'num_ensembles');
disp('Sikeres mentés: tsde_models.mat');