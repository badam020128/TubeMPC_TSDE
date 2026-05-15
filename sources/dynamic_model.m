function x_next = dynamic_model(x, u, params, dt)
    % ODE45 beépített solver a numerikus felrobbanás elkerülésére
    [~, X_res] = ode45(@(t, y) car_dynamics(t, y, u, params), [0, dt], x);
    x_next = X_res(end, :)';
    % Szigorúan rögzítjük a hosszirányú sebességet (Tempomat)
    x_next(4) = params.v_const;
end

function dx = car_dynamics(~, x_curr, u, params)
    psi = x_curr(3);
    v_x = params.v_const; % FIX: Szigorúan konstans sebesség!
    v_y = x_curr(5);
    omega = x_curr(6);
    delta = u(1);

    % Helyes kormányszög előjel (A pozitív delta balra visz)
    alpha_f = delta - atan((v_y + params.l_f * omega) / v_x);
    alpha_r = -atan((v_y - params.l_r * omega) / v_x);

    % Pacejka Magic Formula
    B = params.pacejka_B; C = params.pacejka_C;
    D = params.pacejka_D; E = params.pacejka_E;

    F_yf = D * sin(C * atan(B * alpha_f - E * (B * alpha_f - atan(B * alpha_f))));
    F_yr = D * sin(C * atan(B * alpha_r - E * (B * alpha_r - atan(B * alpha_r))));

    % Keresztirányú és forgó mozgás (Hosszirányú gyorsulás 0)
    dv_x = 0; 
    dv_y = (F_yf * cos(delta) + F_yr) / params.m - omega * v_x;
    domega = (params.l_f * F_yf * cos(delta) - params.l_r * F_yr) / params.I_z;

    dX = v_x * cos(psi) - v_y * sin(psi);
    dY = v_x * sin(psi) + v_y * cos(psi);
    dpsi = omega;

    dx = [dX; dY; dpsi; dv_x; dv_y; domega];
end