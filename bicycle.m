function x_next = bicycle(x, u, v, L, kappa, dt)
    % Bemenetek:
    % x(1) : e_y (keresztirányú pozícióhiba) [m]
    % x(2) : e_psi (irányszög-hiba) [rad]
    % u    : delta (kormányszög) [rad]
    % v    : hosszirányú sebesség [m/s]
    % L    : tengelytáv (wheelbase) [m]
    % kappa: a referenciapálya görbülete [1/m]
    % dt   : mintavételi idő [s]
    
    e_y = x(1);
    e_psi = x(2);
    delta = u;
    
    % Nemlineáris kinematikai egyenletek
    e_y_next = e_y + v * sin(e_psi) * dt;
    e_psi_next = e_psi + (v/L) * tan(delta) * dt - v * kappa * dt;
    
    x_next = [e_y_next; e_psi_next];
end