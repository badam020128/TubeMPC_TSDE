%% ============================================================
%  DATA GATHERING - NOMINAL VS DYNAMIC (data_gather.m)
%  ------------------------------------------------------------
%  Projekt: Tube MPC residual learning (TSDE)
%  Bázis modell: nominal_model.m (Lineáris mátrixok)
%  Valóság: dynamic_model.m (Pacejka dinamika)
%% ============================================================
disp('--- Adatgyűjtés indítása: Nominális vs. Dinamikai modell ---');

% --- Szimulációs és Jármű paraméterek ---
N_data = 50000; % 50k a teszthez, éles tréninghez 500k is ajánlott lehet
dt = 0.1;        % Mintavételi idő [s]
params.v_const = 6.0;  % Sebesség [m/s]
L = 2.5;         % Tengelytáv [m]

% Dinamikai paraméterek a dynamic_model.m-hez (Ground Truth)
params.m = 1500; params.I_z = 3000;
params.l_f = 1.2; params.l_r = 1.3;
params.pacejka_B = 10; params.pacejka_C = 1.9;
params.pacejka_D = 1.0 * (params.m * 9.81 / 2); params.pacejka_E = 0.97;

% 1. NOMINÁLIS MODELL ÉS LQR SZABÁLYOZÓ INICIALIZÁLÁSA
% Megkapjuk az MPC által használt A, B, G mátrixokat
[A, B, G] = nominal_model(params.v_const, L, dt);

% Kiszámolunk egy bázis LQR szabályozót a mintavételezéshez
% Ugyanazokkal a súlyokkal, mint a TubeMPC.m-ben (Q=[10, 5], R=1)
Q_lqr = diag([10, 5]); 
R_lqr = 1;
[K_lqr_mat, ~, ~] = dlqr(A, B, Q_lqr, R_lqr);
K_lqr = -K_lqr_mat; % Kormányszög = K_lqr * x

% --- Adattárolók előkészítése ---
X_train = zeros(2, N_data); % Bemenet: [e_y; e_psi]
U_train = zeros(1, N_data); % Bemenet: [delta] (A tényleges kormányszög!)
K_train = zeros(1, N_data); % Bemenet: [kappa]
R_train = zeros(2, N_data); % Cél (Residual): Dinamikus - Nominális

disp('Intelligens (LQR-vezérelt + Kevert eloszlású) mintavételezés...');

% --- Adatgeneráló Ciklus ---
for i = 1:N_data
    % 1. KEVERT ÁLLAPOT-MINTAVÉTELEZÉS (70% Gauss, 30% Uniform)
    if rand() < 0.7
        % Normál üzem (sűrűbb mintavétel a sáv közepe és kis szögek körül)
        e_y   = 0.5 * randn(); 
        e_psi = 0.1 * randn(); 
        kappa = 0.05 * randn(); 
    else
        % Extrém helyzetek (hogy a háló robusztus maradjon a határokon is)
        e_y   = -4.0 + 8.0 * rand();
        e_psi = -0.5 + 1.0 * rand();
        kappa = -0.222 + 0.444 * rand(); % max_kappa = ~8.0 / v^2 = 0.222
    end
    
    % Biztonsági vágás (Clipping), hogy a randn extrémjei se borítsák fel a Pacejkát
    e_y   = max(min(e_y, 4.0), -4.0);
    e_psi = max(min(e_psi, 0.5), -0.5);
    kappa = max(min(kappa, 0.222), -0.222); 
    
    x_curr = [e_y; e_psi];
    
    % 2. ZÁRT HURKÚ (LQR) KORMÁNYZÁS + GAUSS ZAJ (Exploration)
    delta_ff = L * kappa;                 % Ideális ív kormányszöge (Feedforward)
    delta_fb = K_lqr * x_curr;            % Hiba korrekció (Feedback)
    
    % Gauss-zaj a valósághű zavarások modellezéséhez (Szórás: 0.05 rad)
    delta_noise = 0.05 * randn();         
    
    delta = delta_ff + delta_fb + delta_noise;
    delta = max(-0.5, min(0.5, delta));   % Szaturáció a fizikai korlátra
    
    % 3. NOMINÁLIS LÉPÉS (Lineáris jóslat)
    x_next_nom = A * x_curr + B * delta + G * kappa;
    
    % 4. DINAMIKAI LÉPÉS (A valódi fizika)
    % Kezdeti szögsebesség és kúszásszög beállítása (steady-state cornering)
    omega_init = params.v_const * kappa; 
    beta_approx = params.l_r * kappa; 
    v_y_init = params.v_const * beta_approx; 

    % Kezdőállapot: [X; Y; psi; v_x; v_y; omega]
    x_dyn_init = [0; e_y; e_psi; params.v_const; v_y_init; omega_init];
    
    % Meghívjuk a valós Pacejka dinamikai modellt
    x_dyn_next = dynamic_model(x_dyn_init, [delta; 0], params, dt);
    
    % 5. REFERENCIA PÁLYA ELTOLÓDÁSÁNAK SZÁMÍTÁSA
    ds = params.v_const * dt; 
    psi_ref_next = params.v_const * kappa * dt; % A pálya új irányszöge

    if abs(kappa) > 1e-5
        X_ref_shift = (1/kappa) * sin(kappa * ds);
        Y_ref_shift = (1/kappa) * (1 - cos(kappa * ds));
    else
        X_ref_shift = ds;
        Y_ref_shift = 0;
    end

    % Távolság a globális keretben az ÚJ kanyarodó referenciaponttól
    dx = x_dyn_next(1) - X_ref_shift;
    dy = x_dyn_next(2) - Y_ref_shift;

    % Visszatranszformálás a VALÓS Frenet hibaállapotba (forgatás a pálya szögével)
    e_y_next_true   = -sin(psi_ref_next) * dx + cos(psi_ref_next) * dy; 
    e_psi_next_true = wrapToPi(x_dyn_next(3) - psi_ref_next);

    x_next_true = [e_y_next_true; e_psi_next_true];
    
    % 6. MARADÉK HIBA (RESIDUAL) KISZÁMÍTÁSA
    residual = x_next_true - x_next_nom;
    
    % 7. Adatok eltárolása (A TÉNYLEGES delta kerül mentésre!)
    X_train(:, i) = x_curr;
    U_train(:, i) = delta; 
    K_train(i)    = kappa;
    R_train(:, i) = residual;
    
    if mod(i, 10000) == 0, fprintf('%d/%d pont kész...\n', i, N_data); end
end

% --- Adatok kimentése ---
save('tsde_training_data.mat', 'X_train', 'U_train', 'K_train', 'R_train');
fprintf('Kész! A tisztított adatok elmentve a "tsde_training_data.mat" fájlba.\n');