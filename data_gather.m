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
v_const = 10.0;  % Sebesség [m/s]
L = 2.5;         % Tengelytáv [m]

% Dinamikai paraméterek a dynamic_model.m-hez (Ground Truth)
params.m = 1500; params.I_z = 3000;
params.l_f = 1.2; params.l_r = 1.3;
params.pacejka_B = 10; params.pacejka_C = 1.9;
params.pacejka_D = 1.0; params.pacejka_E = 0.97;

% 1. NOMINÁLIS MODELL INICIALIZÁLÁSA
% Megkapjuk az MPC által használt A, B, G mátrixokat
[A, B, G] = nominal_model(v_const, L, dt);

% --- Adattárolók előkészítése ---
X_train = zeros(2, N_data); % Bemenet: [e_y; e_psi]
U_train = zeros(1, N_data); % Bemenet: [delta]
K_train = zeros(1, N_data); % Bemenet: [kappa]
R_train = zeros(2, N_data); % Cél (Residual): Dinamikus - Nominális

disp('Véletlenszerű mintavételezés a tartományban...');

% --- Adatgeneráló Ciklus ---
for i = 1:N_data
    % 1. Véletlenszerű állapotok és pályagörbület sorsolása
    e_y   = -6.0 + 12.0 * rand();   % e_y   range: [-6, 6] m
    e_psi = -0.8 + 1.6 * rand();    % e_psi range: [-0.8, 0.8] rad
    delta = -0.6 + 1.2 * rand();    % delta range: [-0.6, 0.6] rad
    kappa = -0.4 + 0.8 * rand();    % kappa range: [-0.4, 0.4] 1/m
    
    x_curr = [e_y; e_psi];
    
    % 2. NOMINÁLIS LÉPÉS (Lineáris jóslat)
    % Ez az, amit az "egyszerű" szabályozó várna
    x_next_nom = A * x_curr + B * delta + G * kappa;
    
    % 3. DINAMIKAI LÉPÉS (A valódi fizika)
    % A hibaállapotot globális állapotba transzformáljuk a szimulációhoz
    % Feltételezés: X=0, v_x=v_const, v_y=0 (oldalirányú csúszás nélkül indul)
    x_dyn_init = [0; e_y; e_psi; v_const; 0; 0];
    
    % Meghívjuk a valós dinamikai modellt (gyorsulás a_x = 0)
    x_dyn_next = dynamic_model(x_dyn_init, [delta; 0], params, dt);
    
    % Visszatranszformálás hibaállapotba
    % e_y_next = Y pozíció; e_psi_next = psi - pálya_szöge
    % A pálya iránya dt alatt annyit változik, amennyit a kanyar "fordul": v*kappa*dt
    e_y_next_true   = x_dyn_next(2);
    e_psi_next_true = x_dyn_next(3) - (v_const * kappa * dt);
    x_next_true = [e_y_next_true; e_psi_next_true];
    
    % 4. MARADÉK HIBA (RESIDUAL) KISZÁMÍTÁSA
    % Ezt a "szakadékot" fogja a neurális háló befoltozni
    residual = x_next_true - x_next_nom;
    
    % 5. Adatok eltárolása
    X_train(:, i) = x_curr;
    U_train(:, i) = delta;
    K_train(i)    = kappa;
    R_train(:, i) = residual;
    
    % Progress bar (opcionális)
    if mod(i, 100000) == 0, fprintf('%d/500k pont kész...\n', i); end
end

% --- Adatok kimentése ---
save('tsde_training_data.mat', 'X_train', 'U_train', 'K_train', 'R_train');
fprintf('Kész! A tisztított adatok elmentve a "tsde_training_data.mat" fájlba.\n');