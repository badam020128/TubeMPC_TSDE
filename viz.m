%% ============================================================
%  TRAINING DATA VISUALIZATION & AUDIT (visualize_training_data.m)
%  ------------------------------------------------------------
%  A generált Monte Carlo adathalmaz vizuális ellenőrzése.
%  Feltételezi, hogy a 'smooth_path' és a 'tsde_training_data.mat' létezik.
%% ============================================================
% 1. Adatok betöltése
if ~exist('tsde_training_data.mat', 'file')
    error('Hiba: Nem találom a tsde_training_data.mat fájlt! Futtasd a data_gather.m-et.');
end
load('tsde_training_data.mat'); 
N_total = size(X_train, 2);

% 2. Referenciapálya adatok
x_ref = smooth_path(:, 1);
y_ref = smooth_path(:, 2);
dx = gradient(x_ref);
dy = gradient(y_ref);
psi_ref = atan2(dy, dx);

% 3. Rajzolás
figure('Name', 'Monte Carlo Adathalmaz Bírálata', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 500]);

% --- ELSŐ SZAKASZ: Járművek a pályán (A korábbi Quiver plot) ---
subplot(1,2,1); hold on; axis equal; grid on;
plot(x_ref, y_ref, 'k', 'LineWidth', 2.5, 'DisplayName', 'Referencia Pálya');

% Sávhatárok kirajzolása
max_ey = 6.0; 
plot(x_ref - max_ey.*sin(psi_ref), y_ref + max_ey.*cos(psi_ref), 'r--', 'LineWidth', 1);
plot(x_ref + max_ey.*sin(psi_ref), y_ref - max_ey.*cos(psi_ref), 'r--', 'LineWidth', 1);

% Véletlenszerű minták kiválasztása (max 2000 a sebesség miatt)
N_plot = min(2000, N_total);
idx_s = randperm(N_total, N_plot);
idx_p = randi(length(x_ref), 1, N_plot);

X_p = zeros(1, N_plot); Y_p = zeros(1, N_plot);
U_p = zeros(1, N_plot); V_p = zeros(1, N_plot);

for i = 1:N_plot
    p = idx_p(i); s = idx_s(i);
    X_p(i) = x_ref(p) - X_train(1,s)*sin(psi_ref(p));
    Y_p(i) = y_ref(p) + X_train(1,s)*cos(psi_ref(p));
    psi_glob = psi_ref(p) + X_train(2,s);
    U_p(i) = cos(psi_glob); V_p(i) = sin(psi_glob);
end

quiver(X_p, Y_p, U_p, V_p, 0.5, 'b', 'DisplayName', 'Tanító minták');
xlabel('X [m]'); ylabel('Y [m]');
title('1. Geometriai lefedettség a pálya körül');
legend('Location', 'southoutside', 'Orientation', 'horizontal');

% --- MÁSODIK SZAKASZ: Maradék hiba eloszlása (A "Bírálat" lényege) ---
subplot(1,2,2); hold on; grid on;
% Kiszámoljuk a hiba nagyságát (normáját) minden ponthoz
res_magnitude = vecnorm(R_train);

% Megjelenítjük a hiba eloszlását az e_y és e_psi síkjában
scatter(X_train(1, 1:10:end), X_train(2, 1:10:end), 5, res_magnitude(1:10:end), 'filled');
h = colorbar;
ylabel(h, 'Maradék hiba magnitúdója (m/rad)');
colormap(jet);
xlabel('e_y [m]'); ylabel('e_{\psi} [rad]');
title('2. Hol téved a lineáris modell? (Residual Heatmap)');
view(2);

fprintf('Vizualizáció kész. A jobb oldali ábrán a vörös területek mutatják,\n');
fprintf('ahol a Pacejka-modell már durván eltér a lineáris matematikától.\n');