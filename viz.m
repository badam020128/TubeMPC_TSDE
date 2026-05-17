%% ============================================================
%  TRAINING DATA VISUALIZATION & SIMULATION AUDIT (viz.m)
%  ------------------------------------------------------------
%  1. Monte Carlo adathalmaz vizuális ellenőrzése
%  2. Szimulációs eredmények (AI vs Hagyományos) összehasonlítása
%% ============================================================

% ============================================================
% 1. RÉSZ: TANÍTÓADATOK ÉS HIBA ELOSZLÁS BÍRÁLATA
% ============================================================
if exist('tsde_training_data.mat', 'file') && exist('smooth_path', 'var')
    load('tsde_training_data.mat'); 
    N_total = size(X_train, 2);

    x_ref = smooth_path(:, 1);
    y_ref = smooth_path(:, 2);
    dx = gradient(x_ref);
    dy = gradient(y_ref);
    psi_ref = atan2(dy, dx);

    figure('Name', 'Monte Carlo Adathalmaz Bírálata', 'NumberTitle', 'off', 'Position', [50, 100, 1200, 500]);

    % --- Geometriai lefedettség ---
    subplot(1,2,1); hold on; axis equal; grid on;
    plot(x_ref, y_ref, 'k', 'LineWidth', 2.5, 'DisplayName', 'Referencia Pálya');
    max_ey = 6.0; 
    plot(x_ref - max_ey.*sin(psi_ref), y_ref + max_ey.*cos(psi_ref), 'r--', 'LineWidth', 1);
    plot(x_ref + max_ey.*sin(psi_ref), y_ref - max_ey.*cos(psi_ref), 'r--', 'LineWidth', 1);

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

    % --- Residual Heatmap ---
    subplot(1,2,2); hold on; grid on;
    res_magnitude = vecnorm(R_train);
    scatter(X_train(1, 1:10:end), X_train(2, 1:10:end), 5, res_magnitude(1:10:end), 'filled');
    h = colorbar; ylabel(h, 'Maradék hiba magnitúdója (m/rad)');
    colormap(jet);
    xlabel('e_y [m]'); ylabel('e_{\psi} [rad]');
    title('2. Hol téved a lineáris modell? (Residual Heatmap)');
    view(2);
else
    disp('Nem található a tanítóadat, vagy a smooth_path. Átugrom az 1. részt.');
end


% ============================================================
% 2. RÉSZ: SZIMULÁCIÓS EREDMÉNYEK ÖSSZEHASONLÍTÁSA (AI vs SIMA)
% ============================================================
if exist('results', 'var') && length(results) >= 2
    figure('Name', 'Teljesítmény: AI vs Hagyományos Tube MPC', 'Position', [100, 100, 1400, 800]);

    % --- 1. Pályakövetési hiba (Laterális e_y) ---
    subplot(2, 2, 1); hold on; grid on;
    plot(results(1).e_y, 'r', 'LineWidth', 1.5, 'DisplayName', 'Hagyományos Tube MPC');
    plot(results(2).e_y, 'g', 'LineWidth', 1.5, 'DisplayName', 'AI (TSDE) Tube MPC');
    xlabel('Szimulációs lépés'); ylabel('Laterális hiba (e_y) [m]');
    title('Pályakövetési pontosság');
    legend('Location', 'best');

    % --- 2. Csőméret (d_max) adaptációja ---
    subplot(2, 2, 2); hold on; grid on;
    if isfield(results, 'd_max')
        plot(results(1).d_max, 'r--', 'LineWidth', 2, 'DisplayName', 'Hagyományos d_{max} (Konstans)');
        plot(results(2).d_max, 'g', 'LineWidth', 1.5, 'DisplayName', 'AI d_{max} (Dinamikus bizonytalanság)');
        xlabel('Szimulációs lépés'); ylabel('Cső sugara (d_{max}) [m]');
        title('Robusztussági sáv (Tube) mérete');
        legend('Location', 'best');
    else
        title('Csőméret: Hiányzó adat (Frissítsd a simulate.m-et!)');
    end

    % --- 3. Kormányszög (Beavatkozás simasága) ---
    subplot(2, 2, 3); hold on; grid on;
    if isfield(results, 'u')
        plot(results(1).u, 'r', 'LineWidth', 1.2, 'DisplayName', 'Hagyományos MPC Kormányszög');
        plot(results(2).u, 'g', 'LineWidth', 1.2, 'DisplayName', 'AI MPC Kormányszög');
        xlabel('Szimulációs lépés'); ylabel('\delta Kormányszög [rad]');
        title('Irányítójel analizálása (Rángatás ellenőrzése)');
        legend('Location', 'best');
    else
        title('Kormányszög: Hiányzó adat');
    end

    % --- 4. Trajektória kinagyítva ---
    subplot(2, 2, 4); hold on; grid on; axis equal;
    plot(smooth_path(:,1), smooth_path(:,2), 'k--', 'LineWidth', 2, 'DisplayName', 'Ideális Ív');
    plot(results(1).X, results(1).Y, 'r', 'LineWidth', 1.5, 'DisplayName', 'Hagyományos Tube MPC');
    plot(results(2).X, results(2).Y, 'g', 'LineWidth', 1.5, 'DisplayName', 'AI (TSDE) Tube MPC');
    
    % Fókuszáljunk a pálya egy nehezebb, kanyargósabb részére (dinamikusan keressük meg)
    center_idx = floor(length(results(1).X) * 0.5);
    zoom_window = 25; % méter
    xlim([results(1).X(center_idx) - zoom_window, results(1).X(center_idx) + zoom_window]);
    ylim([results(1).Y(center_idx) - zoom_window, results(1).Y(center_idx) + zoom_window]);
    
    xlabel('X [m]'); ylabel('Y [m]');
    title('Trajektória részlet (Kanyarodás)');
    legend('Location', 'best');

    % ============================================================
    % 3. RÉSZ: PARANCSSORI STATISZTIKAI ÉRTÉKELÉS
    % ============================================================
    fprintf('\n=======================================================\n');
    fprintf('  SZIMULÁCIÓS EREDMÉNYEK (Hagyományos vs. AI Tube MPC)\n');
    fprintf('=======================================================\n');
    
    rmse_sima = sqrt(mean(results(1).e_y.^2));
    rmse_ai   = sqrt(mean(results(2).e_y.^2));
    
    fprintf('Pályakövetési hiba (RMSE):\n');
    fprintf('  - Hagyományos MPC : %.4f m\n', rmse_sima);
    fprintf('  - AI TSDE MPC     : %.4f m\n', rmse_ai);
    fprintf('  -> Teljesítmény javulás: %.2f%%\n\n', (1 - rmse_ai/rmse_sima)*100);
    
    if isfield(results, 'd_max')
        fprintf('Átlagos Tube méret (d_max):\n');
        fprintf('  - Hagyományos MPC : %.4f m (Konzervatív)\n', mean(results(1).d_max));
        fprintf('  - AI TSDE MPC     : %.4f m (Adaptív)\n', mean(results(2).d_max));
        fprintf('  -> Terület nyereség: %.2f%%\n', (1 - mean(results(2).d_max)/mean(results(1).d_max))*100);
    end
    fprintf('=======================================================\n');
else
    fprintf('\nMegjegyzés: Nem találtam szimulációs adatokat (results). Futtasd előbb a simulate.m szkriptet!\n');
end