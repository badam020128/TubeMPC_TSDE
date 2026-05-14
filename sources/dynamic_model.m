function x_next = dynamic_model(x, u, params, dt)
    % Állapotok (x):
    % x(1) : X globális pozíció [m]
    % x(2) : Y globális pozíció [m]
    % x(3) : psi (jármű irányszöge) [rad]
    % x(4) : v_x (hosszirányú sebesség) [m/s]
    % x(5) : v_y (keresztirányú sebesség) [m/s]
    % x(6) : omega (perdület / yaw rate) [rad/s]
    
    % Bemenetek (u):
    % u(1) : delta (első kerék kormányszöge) [rad]
    % u(2) : a_x (hosszirányú gyorsulás/fékezés) [m/s^2]
    
    % Paraméterek (params struct):
    m = params.m;       % Jármű tömege [kg]
    I_z = params.I_z;   % Tehetetlenségi nyomaték Z tengelyre [kg*m^2]
    l_f = params.l_f;   % Súlypont távolsága az első tengelytől [m]
    l_r = params.l_r;   % Súlypont távolsága a hátsó tengelytől [m]
    
    % Pacejka paraméterek (egyszerűsített Magic Formula: F_y = D * sin(C * atan(B * alpha)))
    B = params.pacejka_B; % Stiffness factor
    C = params.pacejka_C; % Shape factor
    D = params.pacejka_D; % Peak value (tapadási határ)
    E = params.pacejka_E; % Curvature factor
    
    % Állapotváltozók kibontása
    psi = x(3);
    v_x = max(x(4), 0.1); % Szingularitás elkerülése kis sebességnél
    v_y = x(5);
    omega = x(6);
    
    delta = u(1);
    a_x = u(2);
    
    %% 1. Csúszási szögek (Slip angles) számítása
    % Első csúszási szög
    alpha_f = atan((v_y + l_f * omega) / v_x) - delta;
    % Hátsó csúszási szög
    alpha_r = atan((v_y - l_r * omega) / v_x);
    
    %% 2. Oldalirányú erők számítása (Pacejka Magic Formula)
    F_yf = D * sin(C * atan(B * alpha_f - E * (B * alpha_f - atan(B * alpha_f))));
    F_yr = D * sin(C * atan(B * alpha_r - E * (B * alpha_r - atan(B * alpha_r))));
    
    %% 3. Jármű dinamikai differenciálegyenletei
    % Hosszirányú gyorsulás (v_x dot) - egyszerűsített kinematikai közelítés a hajtásra
    dv_x = a_x - omega * v_y; 
    
    % Keresztirányú gyorsulás (v_y dot)
    dv_y = (F_yf * cos(delta) + F_yr) / m - omega * v_x;
    
    % Szöggyorsulás (omega dot)
    domega = (l_f * F_yf * cos(delta) - l_r * F_yr) / I_z;
    
    % Globális koordináták differenciálegyenletei
    dX = v_x * cos(psi) - v_y * sin(psi);
    dY = v_x * sin(psi) + v_y * cos(psi);
    dpsi = omega;
    
    %% 4. Numerikus integrálás (Euler-módszer a példa kedvéért, de lehet RK4 is)
    x_next = zeros(6,1);
    x_next(1) = x(1) + dX * dt;
    x_next(2) = x(2) + dY * dt;
    x_next(3) = x(3) + dpsi * dt;
    x_next(4) = x(4) + dv_x * dt;
    x_next(5) = x(5) + dv_y * dt;
    x_next(6) = x(6) + domega * dt;
end