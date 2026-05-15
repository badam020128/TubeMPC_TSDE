%% ============================================================
%  TESZTSZKRIPT: Hagyományos MPC vs. IGAZI TSDE AI-os Tube MPC
%% ============================================================
%clear; clc; % Mindent törlünk az elején

% Biztonsági háló: ha elrontod a rajzolást, kap egy automata tesztpályát
if ~exist('smooth_path', 'var') || isempty(smooth_path)
    disp('Nem sikerült beolvasni a rajzolt pályát! Automatikus tesztpálya generálása...');
    t_default = linspace(0, 2*pi, 300)';
    smooth_path = [50 + 40*cos(t_default), 50 + 20*sin(t_default)];
end

% A modellek betöltése egy struct-ba (Dinamikusan kezeli a darabszámot)
if ~exist('tsde_models.mat', 'file')
    error('Hiba: Nem találom a tsde_models.mat fájlt! Futtasd a TSDE.m-et.');
end
tsde_models = load('tsde_models.mat'); 

% --- Fizikai paraméterek ---
params.dt = 0.1;
params.v_const = 5.0; % Sebesség 5 m/s
params.L = 2.5;
params.m = 1500; params.I_z = 3000;
params.l_f = 1.2; params.l_r = 1.3;
params.pacejka_B = 10; params.pacejka_C = 1.9;
params.pacejka_D = 1.0 * (params.m * 9.81 / 2); 
params.pacejka_E = 0.97;

% --- Pálya újra-mintavételezése a v_const és dt alapján ---
x_raw = smooth_path(:,1); y_raw = smooth_path(:,2);
S_raw = [0; cumsum(sqrt(diff(x_raw).^2 + diff(y_raw).^2))];
s_step = params.v_const * params.dt; 
S_sim = (0 : s_step : S_raw(end))';
x_ref = interp1(S_raw, x_raw, S_sim, 'linear');
y_ref = interp1(S_raw, y_raw, S_sim, 'linear');
psi_ref = unwrap(atan2(gradient(y_ref), gradient(x_ref))); 
kappa_ref = gradient(psi_ref) ./ s_step;

N_sim = length(x_ref) - 20;
Np = 15; % Predikciós horizont

% ========================================================
%  2. SZIMULÁCIÓK FUTTATÁSA
% ========================================================
results = struct();

% Definiáljuk a két futtatást
scenarios = {
    struct('name', 'Sima MPC', 'use_ai', 0, 'd_max', 0.1)    % Lineáris d_max
    struct('name', 'AI Tube MPC', 'use_ai', 1, 'd_max', 0.03) % AI d_max
};

for s = 1:2
    fprintf('\n--- %s Szimuláció indul ---\n', scenarios{s}.name);
    
    % *** KRITIKUS JAVÍTÁS: TubeMPC memóriájának törlése ***
    % Ez garantálja, hogy a CasADi gráf és az előző kormányállás tiszta lappal induljon!
    clear TubeMPC; 
    
    x_curr = [0; 0];
    x_dyn_global = [x_ref(1); y_ref(1); psi_ref(1); params.v_const; 0; 0];
    
    for k = 1:N_sim
        kappa_seq = kappa_ref(k : k+Np-1); % Horizont kivágása
        
        % *** JAVÍTÁS: Csak akkor adjuk át a hálót, ha kell! ***
        if scenarios{s}.use_ai == 1
            models_to_pass = tsde_models;
        else
            models_to_pass = []; % Így a CasADi gráf tiszta és gyors marad!
        end
        
        % TUBE MPC FÜGGVÉNY HÍVÁSA!
        u_applied = TubeMPC(x_curr, kappa_seq, params, models_to_pass, scenarios{s}.use_ai, scenarios{s}.d_max);
        
        % ... többi kód változatlan
        
        % Fizikai léptetés
        x_dyn_global = dynamic_model(x_dyn_global, [u_applied; 0], params, params.dt);
        
        % Hibák visszaszámolása
        dx = x_dyn_global(1) - x_ref(k+1);
        dy = x_dyn_global(2) - y_ref(k+1);
        e_y = -sin(psi_ref(k+1))*dx + cos(psi_ref(k+1))*dy; 
        e_psi = wrapToPi(x_dyn_global(3) - psi_ref(k+1));
        
        x_curr = [e_y; e_psi];
        
        % Adatmentés
        results(s).X(k) = x_dyn_global(1);
        results(s).Y(k) = x_dyn_global(2);
        results(s).e_y(k) = e_y;
    end
end

% ========================================================
% 3. Eredmények kirajzolása
% ========================================================
figure('Name', 'MPC Összehasonlítás', 'Position', [100, 100, 1200, 500]);

% Trajektória
subplot(1, 2, 1); hold on; grid on; axis equal;
plot(x_ref, y_ref, 'k--', 'LineWidth', 2, 'DisplayName', 'Referencia Útvonal');
plot(results(1).X, results(1).Y, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Sima MPC');
plot(results(2).X, results(2).Y, 'g-', 'LineWidth', 2, 'DisplayName', 'AI Tube MPC');
legend('Location', 'best'); 
xlabel('X [m]'); ylabel('Y [m]');
title('Jármű Trajektóriája');

% Oldalirányú hiba
subplot(1, 2, 2); hold on; grid on;
plot(results(1).e_y, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Sima MPC');
plot(results(2).e_y, 'g-', 'LineWidth', 2, 'DisplayName', 'AI Tube MPC');
legend('Location', 'best'); 
xlabel('Szimulációs lépés'); ylabel('Laterális hiba (e_y) [m]');
title('Sávtartási Hiba (e_y)');