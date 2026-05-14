%% ============================================================
%  TSDE TESTING & UNCERTAINTY ANALYSIS (TSDE_test.m)
%  ------------------------------------------------------------
%  A betanított 5+5 háló bírálata: hiba és Tube méret ellenőrzés.
%% ============================================================
clear; clc;
fprintf('--- TSDE Modell bírálat indítása ---\n');

% 1. Adatok és modellek betöltése
if ~exist('tsde_models.mat', 'file') || ~exist('tsde_training_data.mat', 'file')
    error('Hiányzó adatok! Futtasd a data_gather.m és TSDE_train.m scripteket.');
end
load('tsde_models.mat'); % ensemble_stage1, ensemble_stage2, in_settings, tar_settings
load('tsde_training_data.mat'); 

% Teszteljünk egy friss adathalmazon (vagy a tanítóadatok végén)
N_test = 1000;
idx_start = size(X_train, 2) - N_test + 1;
test_inputs = [X_train(:, idx_start:end); U_train(idx_start:end); K_train(idx_start:end)];
true_residuals = R_train(:, idx_start:end);

% 2. PREDIKCIÓ AZ EGYÜTTESSEL (Ensemble)
in_test_norm = mapminmax('apply', test_inputs, in_settings);

% Tárolók az összes háló válaszának
s1_raw = zeros(2, N_test, 5);
s2_raw = zeros(2, N_test, 5);

fprintf('Predikciók futtatása a 10 hálón...\n');
for i = 1:5
    s1_raw(:,:,i) = ensemble_stage1{i}(in_test_norm);
    s2_raw(:,:,i) = ensemble_stage2{i}(in_test_norm);
end

% Stage 1: Átlagos hiba becslése
mean_res_norm = mean(s1_raw, 3);
% Stage 2: Bizonytalanság (Stage 2 átlaga + Stage 1 hálók szórása)
% Ez adja meg a Tube (cső) falait.
uncertainty_norm = mean(s2_raw, 3) + std(s1_raw, 0, 3);

% Visszatranszformálás fizikai mértékegységekre
mean_res = mapminmax('reverse', mean_res_norm, tar_settings);
% Bizonytalanság skálázása (közelítőleg a tartomány fele)
unc_scale = (tar_settings.ymax - tar_settings.ymin) / 2;
uncertainty = uncertainty_norm .* unc_scale;

% 3. STATISZTIKAI BÍRÁLAT
original_error = vecnorm(true_residuals);
remaining_error = vecnorm(true_residuals - mean_res);

d_max_linear = 1.2 * max(original_error);
d_max_tsde   = 1.2 * max(remaining_error);

fprintf('\n=== TELJESÍTMÉNY MUTATÓK ===\n');
fprintf('Átlagos hiba csökkenése:  %.2f%%\n', (1 - mean(remaining_error)/mean(original_error))*100);
fprintf('Eredeti d_max (lineáris): %.4f\n', d_max_linear);
fprintf('Új d_max (TSDE hibrid):   %.4f\n', d_max_tsde);
fprintf('Tube szűkülés mértéke:    %.1f-szeres\n', d_max_linear / d_max_tsde);

% 4. VIZUALIZÁCIÓ
figure('Name', 'TSDE Predikció és Bizonytalanság', 'Position', [100, 100, 1000, 600]);

% --- E_y hiba residual ---
subplot(2,1,1); hold on; grid on;
t = 1:N_test;
% A "Cső" (Tube) kirajzolása
fill([t, fliplr(t)], [mean_res(1,:) + 2*uncertainty(1,:), fliplr(mean_res(1,:) - 2*uncertainty(1,:))], ...
    [0.8 1 0.8], 'EdgeColor', 'none', 'DisplayName', 'TSDE Adaptive Tube (95%)');
plot(true_residuals(1,:), 'k', 'LineWidth', 1, 'DisplayName', 'Valós maradék hiba (Dynamic)');
plot(mean_res(1,:), 'g--', 'LineWidth', 1.5, 'DisplayName', 'TSDE Átlag becslés');
ylabel('e_y residual [m]');
title('Stage 1 & 2: Maradék hiba és bizonytalanság (e_y)');
legend('Location', 'best');

% --- E_psi hiba residual ---
subplot(2,1,2); hold on; grid on;
fill([t, fliplr(t)], [mean_res(2,:) + 2*uncertainty(2,:), fliplr(mean_res(2,:) - 2*uncertainty(2,:))], ...
    [0.8 0.8 1], 'EdgeColor', 'none', 'DisplayName', 'TSDE Adaptive Tube (95%)');
plot(true_residuals(2,:), 'k', 'LineWidth', 1, 'DisplayName', 'Valós maradék hiba (Dynamic)');
plot(mean_res(2,:), 'b--', 'LineWidth', 1.5, 'DisplayName', 'TSDE Átlag becslés');
ylabel('e_{\psi} residual [rad]');
xlabel('Minták száma');
title('Stage 1 & 2: Maradék hiba és bizonytalanság (e_{\psi})');
legend('Location', 'best');