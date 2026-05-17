%% ============================================================
%  TESZTSZKRIPT: Hagyományos MPC vs. IGAZI TSDE AI-os Tube MPC
%% ============================================================
% Biztonsági háló
if ~exist('smooth_path', 'var') || isempty(smooth_path)
    t_default = linspace(0, 2*pi, 300)';
    smooth_path = [50 + 40*cos(t_default), 50 + 20*sin(t_default)];
end

if ~exist('tsde_models.mat', 'file')
    error('Hiba: Nem találom a tsde_models.mat fájlt!');
end
tsde_models = load('tsde_models.mat'); 

% --- Fizikai paraméterek ---
params.dt = 0.1;
params.v_const = 6.0; 
params.L = 2.5;
params.m = 1500; params.I_z = 3000;
params.l_f = 1.2; params.l_r = 1.3;
params.pacejka_B = 10; params.pacejka_C = 1.9;
params.pacejka_D = 1.0 * (params.m * 9.81 / 2); 
params.pacejka_E = 0.97;

% --- Pálya újra-mintavételezése ---
x_raw = smooth_path(:,1); y_raw = smooth_path(:,2);
S_raw = [0; cumsum(sqrt(diff(x_raw).^2 + diff(y_raw).^2))];
s_step = params.v_const * params.dt; 
S_sim = (0 : s_step : S_raw(end))';
x_ref = interp1(S_raw, x_raw, S_sim, 'linear');
y_ref = interp1(S_raw, y_raw, S_sim, 'linear');
psi_ref = unwrap(atan2(gradient(y_ref), gradient(x_ref))); 
kappa_ref = gradient(psi_ref) ./ s_step;

Np = 15; % Predikciós horizont
N_sim = length(x_ref) - 20;

results = struct();
scenarios = {
    struct('name', 'Sima MPC', 'use_ai', 0, 'd_max', 0.1)
    struct('name', 'AI Tube MPC', 'use_ai', 1, 'd_max', 0.03)
};

for s = 1:2
    fprintf('\n--- %s Szimuláció indul ---\n', scenarios{s}.name);
    clear TubeMPC; 
    
    x_curr = [0; 0];
    x_dyn_global = [x_ref(1); y_ref(1); psi_ref(1); params.v_const; 0; 0];
    curr_idx = 1;
    u_prev_sim = 0; % Kormányállás nyilvántartása az AI bemenethez
    ai_residual_filtered = [0; 0];

    for k = 1:N_sim
        search_window_end = min(curr_idx + 50, length(x_ref) - Np);
        search_window = curr_idx : search_window_end;
        
        dist_sq = (x_ref(search_window) - x_dyn_global(1)).^2 + ...
                  (y_ref(search_window) - x_dyn_global(2)).^2;
        [~, local_min_idx] = min(dist_sq);
        curr_idx = search_window(local_min_idx); 
        
        kappa_seq = kappa_ref(curr_idx : curr_idx+Np-1); 
        
        % ==============================================================
        % ÚJ RÉSZ: TSDE HÁLÓZAT KIÉRTÉKELÉSE A CASADI-N KÍVÜL!
        % ==============================================================
        % ==============================================================
        % TSDE HÁLÓZAT KIÉRTÉKELÉSE (PING-PONG ELLENI VÉDELEMMEL)
        % ==============================================================
        if scenarios{s}.use_ai == 1
            % --- JAVÍTÁS 1: Sima kormányszög használata ---
            % Nem a rángatózó u_prev-et adjuk meg, hanem az ideális kanyarodási szöget
            u_steady_state = params.L * kappa_seq(1); 
            nn_in = [x_curr; u_steady_state; kappa_seq(1)];
            in_norm = mapminmax('apply', nn_in, tsde_models.in_settings);
            
            s1_preds = zeros(2, tsde_models.num_ensembles);
            s2_preds = zeros(2, tsde_models.num_ensembles);
            for net_i = 1:tsde_models.num_ensembles
                s1_preds(:, net_i) = tsde_models.ensemble_stage1{net_i}(in_norm);
                s2_preds(:, net_i) = tsde_models.ensemble_stage2{net_i}(in_norm);
            end
            
            % Stage 1: Maradék hiba (Normalizált -> Fizikai)
            res_norm = mean(s1_preds, 2);
            residual_pred = mapminmax('reverse', res_norm, tsde_models.tar_settings);
            
            % Stage 2: Bizonytalanság (Dinamikus d_max)
            unc_norm = mean(s2_preds, 2) + std(s1_preds, 0, 2);
            uncertainty = unc_norm ./ tsde_models.tar_settings.gain;
            
            % AKTIVÁCIÓS KAPU
            severity = x_curr(1)^2 + x_curr(2)^2 + u_steady_state^2 + kappa_seq(1)^2;
            activation_factor = 1.0 - exp(-50.0 * severity);
            
            ai_weight = 1.0 * activation_factor; 
            raw_ai_residual = ai_weight * residual_pred;
            
            % --- JAVÍTÁS 2: Erős aluláteresztő szűrő (Low-Pass Filter) ---
            % Ez akadályozza meg, hogy az AI magas frekvencián tudjon oszcillálni
            alpha_filter = 0.6; % 85% ragaszkodás a régihez, 15% az új jóslat
            ai_residual_filtered = alpha_filter * ai_residual_filtered + (1 - alpha_filter) * raw_ai_residual;
            
            final_ai_residual = ai_residual_filtered;
            
            % A Stage 2 beállítja a cső biztonsági sávját
            current_d_max = max(uncertainty); 
        else
            final_ai_residual = [0; 0];
            ai_residual_filtered = [0; 0]; % Reseteljük a szűrőt
            current_d_max = scenarios{s}.d_max; 
        end
        
        % TUBE MPC FÜGGVÉNY HÍVÁSA AZ AI ZAVARÁSSAL
        u_applied = TubeMPC(x_curr, kappa_seq, params, final_ai_residual, current_d_max);
        u_prev_sim = u_applied;
        
        % Fizikai léptetés
        x_dyn_global = dynamic_model(x_dyn_global, [u_applied; 0], params, params.dt);
        
        % Geometriai hiba
        dx = x_dyn_global(1) - x_ref(curr_idx);
        dy = x_dyn_global(2) - y_ref(curr_idx);
        e_y = -sin(psi_ref(curr_idx))*dx + cos(psi_ref(curr_idx))*dy; 
        e_psi = wrapToPi(x_dyn_global(3) - psi_ref(curr_idx));
        
        x_curr = [e_y; e_psi];
        
        % Adatmentés a cikluson belül
        results(s).X(k) = x_dyn_global(1);
        results(s).Y(k) = x_dyn_global(2);
        results(s).e_y(k) = e_y;
        
        % --- EZT A 3 SORT ADD HOZZÁ ---
        results(s).e_psi(k) = e_psi;
        results(s).d_max(k) = current_d_max; 
        results(s).u(k) = u_applied;
        
    end
end

% Rajzolás
figure('Name', 'MPC Összehasonlítás', 'Position', [100, 100, 1200, 500]);
subplot(1, 2, 1); hold on; grid on; axis equal;
plot(x_ref, y_ref, 'k--', 'LineWidth', 2, 'DisplayName', 'Referencia Útvonal');
plot(results(1).X, results(1).Y, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Sima MPC');
plot(results(2).X, results(2).Y, 'g-', 'LineWidth', 2, 'DisplayName', 'AI Tube MPC');
legend('Location', 'best'); xlabel('X [m]'); ylabel('Y [m]');
title('Jármű Trajektóriája');

subplot(1, 2, 2); hold on; grid on;
plot(results(1).e_y, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Sima MPC');
plot(results(2).e_y, 'g-', 'LineWidth', 2, 'DisplayName', 'AI Tube MPC');
legend('Location', 'best'); xlabel('Szimulációs lépés'); ylabel('Laterális hiba (e_y) [m]');
title('Sávtartási Hiba (e_y)');