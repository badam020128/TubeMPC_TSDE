%% ============================================================
%  DATA GATHERING - NOMINAL VS DYNAMIC (data_gather.m)
%  ------------------------------------------------------------
%  Projekt: Tube MPC residual learning (TSDE)
%  Bázis modell: nominal_model.m (Lineáris mátrixok)
%  Valóság: dynamic_model.m (Pacejka dinamika)
%% ============================================================
disp('--- Adatgyűjtés indítása: Nominális vs. Dinamikai modell ---');

% --- Szimulációs és Jármű paraméterek ---
N_data = 50000; % 500k minta a robusztus tanításhoz
dt = 0.1;        % Mintavételi idő [s]
params.v_const = 10.0;  % Sebesség [m/s]
L = 2.5;         % Tengelytáv [m]

% Dinamikai paraméterek a dynamic_model.m-hez (Ground Truth)
params.m = 1500; params.I_z = 3000;
params.l_f = 1.2; params.l_r = 1.3;
params.pacejka_B = 10; params.pacejka_C = 1.9;
params.pacejka_D = 1.0; params.pacejka_E = 0.97;

% 1. NOMINÁLIS MODELL INICIALIZÁLÁSA
% Megkapjuk az MPC által használt A, B, G mátrixokat
[A, B, G] = nominal_model(params.v_const, L, dt);

% --- Adattárolók előkészítése ---
X_train = zeros(2, N_data); % Bemenet: [e_y; e_psi]
U_train = zeros(1, N_data); % Bemenet: [delta]
K_train = zeros(1, N_data); % Bemenet: [kappa]
R_train = zeros(2, N_data); % Cél (Residual): Dinamikus - Nominális

disp('Véletlenszerű mintavételezés a tartományban...');

% --- Adatgeneráló Ciklus ---
for i = 1:N_data
    % 1. Véletlenszerű állapotok és pályagörbület sorsolása
    e_y   = -6.0 + 12.0 * rand();   
    e_psi = -0.8 + 1.6 * rand();    
    delta = -0.6 + 1.2 * rand();    
    kappa = -0.4 + 0.8 * rand();    
    
    x_curr = [e_y; e_psi];
    
    % 2. NOMINÁLIS LÉPÉS (Lineáris jóslat)
    x_next_nom = A * x_curr + B * delta + G * kappa;
    
    % 3. DINAMIKAI LÉPÉS (A valódi fizika)
    % JAVÍTÁS 1: Kezdeti szögsebesség megadása (steady-state cornering)
    omega_init = params.v_const * kappa; 
    
    % JAVÍTOTT kód a data_gather.m-ben:
    beta_approx = params.l_r * kappa; % Kúszásszög közelítés
    v_y_init = params.v_const * beta_approx; 

    % Kezdőállapot: [X; Y; psi; v_x; v_y; omega]
    x_dyn_init = [0; e_y; e_psi; params.v_const; v_y_init; omega_init];;
    
    % Meghívjuk a valós Pacejka dinamikai modellt
    x_dyn_next = dynamic_model(x_dyn_init, [delta; 0], params, dt);
    
    % JAVÍTÁS 2: A referenciapálya eltolódásának figyelembevétele
    % (Az út kanyarodik el a kocsi alól, nem csak a kocsi mozog)
    ds = params.v_const * dt; 
    if abs(kappa) > 1e-5
        Y_ref_shift = (1/kappa) * (1 - cos(kappa * ds));
    else
        Y_ref_shift = 0;
    end
    
    % Visszatranszformálás hibaállapotba
    e_y_next_true   = x_dyn_next(2) - Y_ref_shift; 
    e_psi_next_true = x_dyn_next(3) - (params.v_const * kappa * dt);
    x_next_true = [e_y_next_true; e_psi_next_true];
    
    % 4. MARADÉK HIBA (RESIDUAL) KISZÁMÍTÁSA
    % Ezt a tiszta fizikát fogja tanulni az AI
    residual = x_next_true - x_next_nom;
    
    % 5. Adatok eltárolása
    X_train(:, i) = x_curr;
    U_train(:, i) = delta;
    K_train(i)    = kappa;
    R_train(:, i) = residual;
    
    if mod(i, 10000) == 0, fprintf('%d/50000 pont kész...\n', i); end
end

% --- Adatok kimentése ---
save('tsde_training_data.mat', 'X_train', 'U_train', 'K_train', 'R_train');
fprintf('Kész! A tisztított adatok elmentve a "tsde_training_data.mat" fájlba.\n');