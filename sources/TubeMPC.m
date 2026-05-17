function u_applied = TubeMPC(x_meas, kappa_horizon, params, ai_residual, d_max)
    % IGAZI TUBE MPC - Tiszta Lineáris Predikció + AI Feedforward
    
    % A x_nom_prev rögzíti a nominális predikciót a lépések között
    persistent opti vars K_lqr A_ca B_ca u_prev_val x_nom_prev
    
    import casadi.*
    Np = 15; % Predikciós horizont
    max_ey = 3.0; % Útpálya szélessége (maximális megengedett laterális hiba [m])
    dU_max = 0.2; % Kormányzási sebesség korlát: max 0.3 rad / lépés
    
    if isempty(u_prev_val)
        u_prev_val = 0;
    end
    
    % Kezdeti nominális állapot
    if isempty(x_nom_prev)
        x_nom_prev = x_meas;
    end
    
    % =====================================================================
    % 1. INICIALIZÁLÁS (Csak a legelső futáskor épül fel a szimbolikus gráf)
    % =====================================================================
    if isempty(opti)
        fprintf('--- Stabilizált Tube MPC Inicializálása (CasADi) ---\n');
        opti = Opti();
        
        [A, B, ~] = nominal_model(params.v_const, params.L, params.dt);
        A_ca = DM(A); B_ca = DM(B);
        
        Q_lqr = diag([10, 5]); R_lqr = 1;
        [K_lqr_mat, ~, ~] = dlqr(A, B, Q_lqr, R_lqr);
        K_lqr = DM(-K_lqr_mat);
        rho = max(abs(eig(A - B * K_lqr_mat))); 
        
        % --- Változók ---
        vars.X = opti.variable(2, Np+1);
        vars.U = opti.variable(1, Np);
        vars.S = opti.variable(1, Np+1); 
        
        % --- Paraméterek ---
        vars.x_meas     = opti.parameter(2, 1);
        vars.x_nom_prev = opti.parameter(2, 1); 
        vars.kappa      = opti.parameter(1, Np);
        vars.p_ai_res   = opti.parameter(2, 1); % <--- ÚJ: Az AI által jósolt fix hiba
        vars.p_dmax     = opti.parameter(1, 1);
        vars.u_prev     = opti.parameter(1, 1); 
        
        % --- Célfüggvény és kényszerek ---
        % Mivel az AI stabil, visszatérhetünk normális súlyokra:
        Q_mpc = DM(diag([50, 20])); % Erős sávtartás
        R_mpc = DM(20);             % Finom kormányszög büntetés
        R_dU_mpc = DM(200);           % Normális kormányszervó
        J = 0;
        
        for k = 1:Np
            x_k = vars.X(:,k); u_k = vars.U(k); kap_k = vars.kappa(k);
            
            % --- MEGOLDÁS: A zavarás exponenciális csillapítása a horizonton ---
            decay_rate = 0.5; % Lépésenként feleződik a jósolt hiba hatása
            current_ai_res = vars.p_ai_res * (decay_rate^(k-1));
            
            % TISZTA lineáris lépés + csillapodó zavarás
            x_lin = A_ca * x_k + B_ca * u_k + [0; -params.v_const * kap_k * params.dt] + current_ai_res;
            
            % Dinamikai kényszer
            opti.subject_to(vars.X(:,k+1) == x_lin);
            
            % Cső (Tube) kényszerek
            opti.subject_to(vars.S(k+1) == rho^2 * vars.S(k) + vars.p_dmax^2);

            S_safe = fmax(vars.S(k+1), 1e-6);
            tightening = fmin(sqrt(S_safe), max_ey - 0.2);

            opti.subject_to(vars.X(1, k+1) <=  max_ey - tightening);
            opti.subject_to(vars.X(1, k+1) >= -max_ey + tightening);
            
            % Beavatkozó korlátok
            opti.subject_to(u_k >= -0.5); 
            opti.subject_to(u_k <= 0.5);
            
            if k == 1
                dU = u_k - vars.u_prev;
            else
                dU = u_k - vars.U(k-1);
            end
            opti.subject_to(dU >= -dU_max);
            opti.subject_to(dU <=  dU_max);
            
            J = J + x_k' * Q_mpc * x_k + u_k * R_mpc * u_k + dU * R_dU_mpc * dU;
        end
        opti.minimize(J);
        
        opti.subject_to(vars.X(:,1) == vars.x_nom_prev);
        opti.subject_to(vars.S(1) == 1e-4);
        
        opts = struct();
        opts.ipopt.print_level = 0;
        opts.print_time = 0;
        opts.ipopt.hessian_approximation = 'limited-memory'; 
        opts.ipopt.max_iter = 100;
        opts.ipopt.tol = 1e-3;
        
        opti.solver('ipopt', opts);
    end
    
    % =====================================================================
    % 2. FUTTATÁS (Minden szimulációs lépésben)
    % =====================================================================
    opti.set_value(vars.x_meas, x_meas(:)); 
    
    % Filter
    alpha_nom = 0.8; 
    x_nom_blended = alpha_nom * x_nom_prev(:) + (1 - alpha_nom) * x_meas(:);
    
    opti.set_value(vars.x_nom_prev, x_nom_blended); 
    opti.set_value(vars.kappa,  kappa_horizon(:)'); 
    opti.set_value(vars.p_ai_res, ai_residual(:)); % <--- Átadjuk a háló jóslatát
    opti.set_value(vars.p_dmax, d_max);
    opti.set_value(vars.u_prev, u_prev_val);
    
    try
        sol = opti.solve();
        
        u_opt = sol.value(vars.U(1));
        x_nom_opt = sol.value(vars.X(:,1));
        
        u_applied = u_opt + full(K_lqr * (x_meas(:) - x_nom_opt(:)));
        
        % Warm start
        opti.set_initial(vars.U, [sol.value(vars.U(2:end)), 0]);
        opti.set_initial(vars.X, [sol.value(vars.X(:, 2:end)), sol.value(vars.X(:, end))]);
        opti.set_initial(vars.S, [sol.value(vars.S(2:end)), sol.value(vars.S(end))]);
        
        x_nom_prev = sol.value(vars.X(:,2));
        
    catch
        opti.set_initial(vars.U, zeros(1, Np));
        opti.set_initial(vars.X, repmat(x_meas(:), 1, Np+1));
        opti.set_initial(vars.S, 1e-4 * ones(1, Np+1));
        
        x_nom_prev = x_meas(:);
        
        % LQR Fallback
        u_applied = full(K_lqr * x_meas) + params.L * kappa_horizon(1);
    end
    
    u_applied = max(-0.5, min(0.5, u_applied));
    u_applied = max(u_prev_val - dU_max, min(u_prev_val + dU_max, u_applied));
    
    u_prev_val = u_applied;
end