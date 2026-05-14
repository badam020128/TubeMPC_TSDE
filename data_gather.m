%% ============================================================
%  DATA GATHERING SCRIPT FOR TSDE (data_gather.m)
%  ------------------------------------------------------------
%  Véletlenszerű mintavételezés a teljes állapottéren, hogy a 
%  TSDE háló (ensemble) a fizikai határokon belül mindenhol 
%  pontosan tudja becsülni a lineáris és nemlineáris modell 
%  közötti maradék hibát (residual).
%% ============================================================
disp('--- Adatgyűjtés indítása a TSDE neurális hálóhoz ---');

% --- Rendszer és Szimulációs paraméterek ---
N_data = 500000; % Minták száma (a papír alapján elegendő)
dt = 0.1;       % Mintavételi idő [s]
v = 10.0;       % Hosszirányú sebesség [m/s]
L = 2.5;        % Tengelytáv [m]

% --- Operációs tartományok (Operational Domain) ---
% A tanulmány alapján beállított minimum-maximum értékek
e_y_range   = [-6.0, 6.0];   % Keresztirányú hiba [m]
e_psi_range = [-0.8, 0.8];   % Irányszöghiba [rad]
delta_range = [-0.6, 0.6];   % Kormányszög [rad]
kappa_range = [-0.4, 0.4];   % Pályagörbület [1/m]

% --- Nominális Lineáris Modell Mátrixai ---
A_lin = [1,  v*dt; 
         0,  1];
B_lin = [0; 
         (v/L)*dt];

% --- Adattárolók előkészítése ---
X_train = zeros(2, N_data); % Állapotok: [e_y; e_psi]
U_train = zeros(1, N_data); % Bemenet: [delta]
K_train = zeros(1, N_data); % Görbület (kiegészítő input hálónak)
R_train = zeros(2, N_data); % Célváltozó (Residual error)

disp('Állapottér véletlenszerű mintavételezése...');

% --- Adatgeneráló Ciklus ---
for i = 1:N_data
    % 1. Véletlenszerű állapotok sorsolása a megadott határokon belül
    e_y   = e_y_range(1)   + (e_y_range(2)   - e_y_range(1))   * rand();
    e_psi = e_psi_range(1) + (e_psi_range(2) - e_psi_range(1)) * rand();
    delta = delta_range(1) + (delta_range(2) - delta_range(1)) * rand();
    kappa = kappa_range(1) + (kappa_range(2) - kappa_range(1)) * rand();
    
    x_bic = [e_y; e_psi];
    
    % 2. Valós nemlineáris ugrás kiszámítása a bicycle.m külső függvénnyel
    x_next_true = bicycle(x_bic, delta, v, L, kappa, dt);
    
    % 3. Nominális (lineáris) lépés kiszámítása
    x_next_linear = A_lin * x_bic + B_lin * delta + [0; -v * kappa * dt];
    
    % 4. Maradék hiba (Residual)
    residual = x_next_true - x_next_linear;
    
    % 5. Adatok eltárolása a TSDE számára
    X_train(:, i) = x_bic;
    U_train(:, i) = delta;
    K_train(i)    = kappa;
    R_train(:, i) = residual;
end

% --- Adatok kimentése fájlba ---
save('tsde_training_data.mat', 'X_train', 'U_train', 'K_train', 'R_train');
fprintf('Adatgyűjtés befejezve. %d minta sikeresen elmentve a "tsde_training_data.mat" fájlba.\n', N_data);

%% ÁBRÁZOLÁS
%% ============================================================
%  MONTE CARLO ADATOK VIZUALIZÁCIÓJA A PÁLYÁN
%  ------------------------------------------------------------
%  Feltételezi, hogy a path_define.m és a data_gather.m 
%  már lefutott, így a 'smooth_path' és az adatok elérhetőek.
%% ============================================================
clearvars -except smooth_path; 
clc;

% 1. Adatok betöltése
load('tsde_training_data.mat'); % Betölti: X_train, U_train, K_train, R_train
N_data = size(X_train, 2);

% 2. Referenciapálya geometriájának kinyerése
x_ref = smooth_path(:, 1);
y_ref = smooth_path(:, 2);
dx = gradient(x_ref);
dy = gradient(y_ref);
psi_ref = atan2(dy, dx); % Pálya érintőjének (orrirányának) szöge

% 3. Rajzolás előkészítése
figure('Name', 'Monte Carlo Adatgenerálás Bírálata', 'NumberTitle', 'off', 'Position', [100, 100, 900, 700]);
hold on; axis equal; grid on;
xlabel('X [m]'); ylabel('Y [m]');
title('Generált tanítóadatok eloszlása a referenciapálya körül');

% Referenciapálya kirajzolása
plot(x_ref, y_ref, 'k', 'LineWidth', 3, 'DisplayName', 'Referencia Pálya');

% Operációs tartomány határainak (Tube) kirajzolása (pl. ±6 méter)
max_ey = 6.0; % A data_gather.m-ben beállított e_y_range határa
bound_left_x = x_ref - max_ey .* sin(psi_ref);
bound_left_y = y_ref + max_ey .* cos(psi_ref);
bound_right_x = x_ref - (-max_ey) .* sin(psi_ref);
bound_right_y = y_ref + (-max_ey) .* cos(psi_ref);

plot(bound_left_x, bound_left_y, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Operációs tartomány (+6m)');
plot(bound_right_x, bound_right_y, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Operációs tartomány (-6m)');

% 4. Véletlenszerű adatpontok kiválasztása a plotoláshoz 
% (Nem rajzoljuk ki mind az 50 ezret, hogy ne fagyjon le a gép)
N_plot = 1500; 
idx_samples = randi(N_data, 1, N_plot);            % Melyik adatpontokat rajzoljuk
idx_path    = randi(length(x_ref), 1, N_plot);     % Melyik pályapont mellé vetítjük

X_plot = zeros(1, N_plot);
Y_plot = zeros(1, N_plot);
U_plot = zeros(1, N_plot); % Irányvektor X
V_plot = zeros(1, N_plot); % Irányvektor Y

for i = 1:N_plot
    p_idx = idx_path(i);
    s_idx = idx_samples(i);

    % A háló által látott hibaállapotok
    e_y = X_train(1, s_idx);
    e_psi = X_train(2, s_idx);

    % Visszavetítés Globális X-Y koordinátarendszerbe
    X_plot(i) = x_ref(p_idx) - e_y * sin(psi_ref(p_idx));
    Y_plot(i) = y_ref(p_idx) + e_y * cos(psi_ref(p_idx));

    % Jármű globális orriránya
    psi_global = psi_ref(p_idx) + e_psi;
    U_plot(i) = cos(psi_global);
    V_plot(i) = sin(psi_global);
end

% Nyilak rajzolása (Járművek pozíciója és irányszöge)
quiver(X_plot, Y_plot, U_plot, V_plot, 0.4, 'b', 'MaxHeadSize', 1, 'DisplayName', 'Jármű állapotok (e_y, e_{\psi})');

legend('Location', 'best');
disp('Vizualizáció kész! A kék nyilak mutatják, hogy a háló milyen pozíciókból és szögekből tanulja a fizikát.');