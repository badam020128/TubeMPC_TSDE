function u_applied = TubeMPC(x_meas, kappa_horizon, params, tsde_models, use_ai, d_max)
    % Egységes Tube MPC Szabályozó (Natív CasADi + Nyers TSDE Neurális Háló)
    % Bemenetek:
    %   x_meas        : [e_y; e_psi] aktuális mért hibaállapot
    %   kappa_horizon : Jövőbeli görbületi profil (Np hosszan)
    %   params        : Fizikai paraméterek (v_const, L, dt)
    %   tsde_models   : Betanított AI modellek structja
    %   use_ai        : 0 (Hagyományos MPC) vagy 1 (AI-os Tube MPC)
    %   d_max         : A bizonytalansági cső méretezéséhez szükséges hiba korlát
    
    persistent opti vars nn_data K_lqr A_ca B_ca
    
    import casadi.*
    Np = 15; % Predikciós horizont
    
    % =====================================================================
    % 1. INICIALIZÁLÁS (Csak a legelső futáskor épül fel a CasADi gráf!)
    % =====================================================================
    if isempty(opti)
        fprintf('--- Tube MPC Inicializálása (CasADi gráf építése) ---\n');
        opti = Opti();
        
        [A, B, ~] = nominal_model(params.v_const, params.L, params.dt);
        A_ca = DM(A); B_ca = DM(B);
        
        % LQR tervezés a visszacsatoláshoz és a Tube méretezéshez
        Q_lqr = diag([10, 5]); R_lqr = 1;
        [K_lqr_mat, P_tube_mat, ~] = dlqr(A, B, Q_lqr, R_lqr);
        K_lqr = DM(-K_lqr_mat);
        P_tube = DM(P_tube_mat);
        rho = max(abs(eig(A - B * K_lqr_mat))); % Stabilitási sugár
        
        % AI Mátrixok kicsomagolása (Ha van AI)
        nn_data = struct();
        if ~isempty(tsde_models)
            nn_data.in_off = DM(tsde_models.in_settings.xoffset);
            nn_data.in_g   = DM(tsde_models.in_settings.gain);
            nn_data.in_ymin= DM(tsde_models.in_settings.ymin);
            nn_data.tar_off= DM(tsde_models.tar_settings.xoffset);
            nn_data.tar_g  = DM(tsde_models.tar_settings.gain);
            nn_data.tar_ymin= DM(tsde_models.tar_settings.ymin);
            
            for i = 1:5
                nn_data.W1{i} = DM(tsde_models.ensemble_stage1{i}.IW{1,1});
                nn_data.b1{i} = DM(tsde_models.ensemble_stage1{i}.b{1});
                nn_data.W2{i} = DM(tsde_models.ensemble_stage1{i}.LW{2,1});
                nn_data.b2{i} = DM(tsde_models.ensemble_stage1{i}.b{2});
                nn_data.W3{i} = DM(tsde_models.ensemble_stage1{i}.LW{3,2});
                nn_data.b3{i} = DM(tsde_models.ensemble_stage1{i}.b{3});
            end
        end
        
        % --- CASADI Változók és Paraméterek ---
        vars.X = opti.variable(2, Np+1);
        vars.U = opti.variable(1, Np);
        vars.S = opti.variable(1, Np+1); % Cső (Tube) mérete
        
        vars.x_meas = opti.parameter(2, 1);
        vars.kappa  = opti.parameter(1, Np);
        vars.p_ai   = opti.parameter(1, 1); % AI be/ki kapcsoló (0 vagy 1)
        vars.p_dmax = opti.parameter(1, 1); % Aktuális d_max
        
        % --- Költségfüggvény és Dinamika ---
        Q_mpc = DM(diag([10, 1])); R_mpc = DM(1);
        J = 0;
        
        for k = 1:Np
            x_k = vars.X(:,k); u_k = vars.U(k); kap_k = vars.kappa(k);
            
            % Lineáris alap dinamika
            x_lin = A_ca * x_k + B_ca * u_k + [0; -params.v_const * kap_k * params.dt];
            
            % Neurális háló kiértékelése
            if ~isempty(tsde_models)
                nn_in = [x_k; u_k; kap_k];
                in_norm = (nn_in - nn_data.in_off) .* nn_data.in_g + nn_data.in_ymin;
                res_norm = 0;
                for i = 1:5
                    a1 = tanh(nn_data.W1{i} * in_norm + nn_data.b1{i});
                    a2 = tanh(nn_data.W2{i} * a1      + nn_data.b2{i});
                    res_norm = res_norm + (nn_data.W3{i} * a2 + nn_data.b3{i});
                end
                res_norm = res_norm / 5;
                residual = (res_norm - nn_data.tar_ymin) ./ nn_data.tar_g + nn_data.tar_off;
            else
                residual = [0; 0];
            end
            
            % DINAMIKAI KÉNYSZER (Itt adjuk hozzá az AI-t a p_ai szorzóval!)
            opti.subject_to(vars.X(:,k+1) == x_lin + vars.p_ai * residual);
            
            % TUBE Növekedési Kényszer
            opti.subject_to(vars.S(k+1) == rho^2 * vars.S(k) + vars.p_dmax^2);
            
            % Kormányzási korlátok
            opti.subject_to(u_k >= -0.5); opti.subject_to(u_k <= 0.5);
            
            % Költség adása
            J = J + x_k' * Q_mpc * x_k + u_k * R_mpc * u_k;
        end
        % Végső költség (Terminal cost)
        J = J + vars.X(:,Np+1)' * P_tube * vars.X(:,Np+1);
        opti.minimize(J);
        
        % Kezdeti feltételek
        opti.subject_to(vars.X(:,1) == vars.x_meas);
        opti.subject_to(vars.S(1) == 1e-4); % Kezdeti csőméret
        
        opti.solver('ipopt', struct('ipopt', struct('print_level', 0), 'print_time', 0));
    end
    
    % =====================================================================
    % 2. FUTTATÁS (Minden szimulációs lépésben ez fut le)
    % =====================================================================
    opti.set_value(vars.x_meas, x_meas);
    opti.set_value(vars.kappa, kappa_horizon);
    opti.set_value(vars.p_ai, use_ai);     % AI bekapcsolása (1) vagy kikapcsolása (0)
    opti.set_value(vars.p_dmax, d_max);    % Adott módhoz tartozó d_max
    
    try
        sol = opti.solve();
        u_opt = sol.value(vars.U(1));
        
        % Warm-start a következő lépéshez (Ettől lesz valós idejű a CasADi!)
        opti.set_initial(vars.U, [sol.value(vars.U(2:end)), 0]);
        opti.set_initial(vars.X, [sol.value(vars.X(:, 2:end)), sol.value(vars.X(:, end))]);
        opti.set_initial(vars.S, [sol.value(vars.S(2:end)), sol.value(vars.S(end))]);
        
        % Visszacsatolás (Tube MPC control law: u = v + K*e, e = 0 mert x_nom=x_meas indításkor)
        u_applied = u_opt;
    catch
        % Fallback LQR ha infeasible lenne
        u_applied = full(K_lqr * x_meas);
    end
    
    u_applied = max(-0.5, min(0.5, u_applied)); % Hard limit
end