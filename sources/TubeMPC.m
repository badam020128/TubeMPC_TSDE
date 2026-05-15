function u_applied = TubeMPC(x_meas, kappa_horizon, params, tsde_models, use_ai, d_max)
    % Egységes Tube MPC Szabályozó
    % JAVÍTVA: L-BFGS Hessian, Slew Rate Constraint, LQR Feedforward, Dinamikus AI méret
    %
    % Bemenetek:
    %   x_meas        : [e_y; e_psi] aktuális mért hibaállapot
    %   kappa_horizon : Jövőbeli görbületi profil (Np hosszan)
    %   params        : Fizikai paraméterek (v_const, L, dt)
    %   tsde_models   : Betanított AI modellek structja (ha Sima MPC, lehet [])
    %   use_ai        : 0 (Hagyományos) vagy 1 (AI-os Tube MPC)
    %   d_max         : A bizonytalansági cső méretezéséhez szükséges hiba korlát
    
    % Hozzáadtuk az u_prev_val változót a kormánysebesség (slew rate) nyomonkövetéséhez
    persistent opti vars nn_data K_lqr A_ca B_ca u_prev_val
    
    import casadi.*
    Np = 15; % Predikciós horizont
    max_ey = 3.0; % Útpálya szélessége (maximális megengedett laterális hiba [m])
    dU_max = 0.1; % Kormányzási sebesség korlát: max 0.1 rad / lépés (kb. 5.7 fok)
    
    % Ha a memóriatér üres (első lépés), a kormány is nullán áll
    if isempty(u_prev_val)
        u_prev_val = 0;
    end
    
    % =====================================================================
    % 1. INICIALIZÁLÁS (Csak a legelső futáskor épül fel a szimbolikus gráf)
    % =====================================================================
    if isempty(opti)
        fprintf('--- Tube MPC Inicializálása (CasADi gráf építése) ---\n');
        opti = Opti();
        
        [A, B, ~] = nominal_model(params.v_const, params.L, params.dt);
        A_ca = DM(A); B_ca = DM(B);
        
        % LQR tervezés a stabilitási sugár (rho) és a Fallback miatt
        Q_lqr = diag([10, 5]); R_lqr = 1;
        [K_lqr_mat, ~, ~] = dlqr(A, B, Q_lqr, R_lqr);
        K_lqr = DM(-K_lqr_mat);
        rho = max(abs(eig(A - B * K_lqr_mat))); 
        
        % AI paraméterek betöltése dinamikusan (csak ha tényleg AI-t futtatunk)
        nn_data = struct();
        if ~isempty(tsde_models) && use_ai == 1
            % Kiolvassuk, hány háló van a modellben
            nn_data.num_nets = length(tsde_models.ensemble_stage1);
            
            nn_data.in_off = DM(tsde_models.in_settings.xoffset);
            nn_data.in_g   = DM(tsde_models.in_settings.gain);
            nn_data.in_ymin= DM(tsde_models.in_settings.ymin);
            nn_data.tar_off= DM(tsde_models.tar_settings.xoffset);
            nn_data.tar_g  = DM(tsde_models.tar_settings.gain);
            nn_data.tar_ymin= DM(tsde_models.tar_settings.ymin);
            
            for i = 1:nn_data.num_nets
                nn_data.W1{i} = DM(tsde_models.ensemble_stage1{i}.IW{1,1});
                nn_data.b1{i} = DM(tsde_models.ensemble_stage1{i}.b{1});
                nn_data.W2{i} = DM(tsde_models.ensemble_stage1{i}.LW{2,1});
                nn_data.b2{i} = DM(tsde_models.ensemble_stage1{i}.b{2});
                nn_data.W3{i} = DM(tsde_models.ensemble_stage1{i}.LW{3,2});
                nn_data.b3{i} = DM(tsde_models.ensemble_stage1{i}.b{3});
            end
        end
        
        % --- Változók ---
        vars.X = opti.variable(2, Np+1);
        vars.U = opti.variable(1, Np);
        vars.S = opti.variable(1, Np+1); % Csőméret négyzete
        
        % --- Paraméterek ---
        vars.x_meas = opti.parameter(2, 1);
        vars.kappa  = opti.parameter(1, Np);
        vars.p_ai   = opti.parameter(1, 1);
        vars.p_dmax = opti.parameter(1, 1);
        vars.u_prev = opti.parameter(1, 1); % Előző lépés kormányállása
        
        % --- Optimalizációs célfüggvény és kényszerek ---
        Q_mpc = DM(diag([20, 2])); 
        R_mpc = DM(0.5);
        R_dU_mpc = DM(5.0); % Erősen büntetjük a kormányzási rángatást!
        J = 0;
        
        for k = 1:Np
            x_k = vars.X(:,k); u_k = vars.U(k); kap_k = vars.kappa(k);
            
            % 1. Névleges dinamika
            x_lin = A_ca * x_k + B_ca * u_k + [0; -params.v_const * kap_k * params.dt];
            
            % 2. AI Residual korrekció
            if ~isempty(tsde_models) && use_ai == 1
                nn_in = [x_k; u_k; kap_k];
                in_norm = (nn_in - nn_data.in_off) .* nn_data.in_g + nn_data.in_ymin;
                res_norm = 0;
                % Dinamikus ciklus a hálózatok darabszáma alapján
                for i = 1:nn_data.num_nets
                    a1 = tanh(nn_data.W1{i} * in_norm + nn_data.b1{i});
                    a2 = tanh(nn_data.W2{i} * a1      + nn_data.b2{i});
                    res_norm = res_norm + (nn_data.W3{i} * a2 + nn_data.b3{i});
                end
                res_norm = res_norm / nn_data.num_nets;
                residual = (res_norm - nn_data.tar_ymin) ./ nn_data.tar_g + nn_data.tar_off;
            else
                residual = [0; 0];
            end
            
            % Dinamikai kényszer az AI-val korrigálva
            opti.subject_to(vars.X(:,k+1) == x_lin + vars.p_ai * residual);
            
            % TUBE dinamika (bizonytalanság terjedése)
            opti.subject_to(vars.S(k+1) == rho^2 * vars.S(k) + vars.p_dmax^2);

            % Biztonságos Constraint Tightening (A végtelen gyök derivált kivédése)
            S_safe = fmax(vars.S(k+1), 1e-6);
            tightening = fmin(sqrt(S_safe), max_ey - 0.2);

            opti.subject_to(vars.X(1, k+1) <=  max_ey - tightening);
            opti.subject_to(vars.X(1, k+1) >= -max_ey + tightening);
            
            % Bemeneti abszolút korlátok (Kormányszög max 0.5 rad)
            opti.subject_to(u_k >= -0.5); 
            opti.subject_to(u_k <= 0.5);
            
            % Kormánysebesség (Slew Rate) korlát és dU kiszámítása
            if k == 1
                dU = u_k - vars.u_prev;
            else
                dU = u_k - vars.U(k-1);
            end
            opti.subject_to(dU >= -dU_max);
            opti.subject_to(dU <=  dU_max);
            
            % Célfüggvény bővítése a dU büntetéssel
            J = J + x_k' * Q_mpc * x_k + u_k * R_mpc * u_k + dU * R_dU_mpc * dU;
        end
        opti.minimize(J);
        
        % Kezdeti feltételek
        opti.subject_to(vars.X(:,1) == vars.x_meas);
        opti.subject_to(vars.S(1) == 1e-4);
        
        % Solver beállítások
        opts = struct();
        opts.ipopt.print_level = 0;
        opts.print_time = 0;
        opts.ipopt.hessian_approximation = 'limited-memory'; % <--- A háló miatt kötelező!
        opts.ipopt.max_iter = 100;
        
        opti.solver('ipopt', opts);
    end
    
    % =====================================================================
    % 2. FUTTATÁS (Minden szimulációs lépésben)
    % =====================================================================
    opti.set_value(vars.x_meas, x_meas(:)); 
    opti.set_value(vars.kappa,  kappa_horizon(:)'); 
    opti.set_value(vars.p_ai,   use_ai);
    opti.set_value(vars.p_dmax, d_max);
    opti.set_value(vars.u_prev, u_prev_val); % Aktuális kormányállás átadása
    
    try
        sol = opti.solve();
        u_opt = sol.value(vars.U(1));
        
        % Warm-start a következő lépéshez (Segít a gyorsaságban)
        opti.set_initial(vars.U, [sol.value(vars.U(2:end)), 0]);
        opti.set_initial(vars.X, [sol.value(vars.X(:, 2:end)), sol.value(vars.X(:, end))]);
        opti.set_initial(vars.S, [sol.value(vars.S(2:end)), sol.value(vars.S(end))]);
        
        u_applied = u_opt;
    catch
        % Fallback LQR-re, ha a solver nem talál megoldást
        % JAVÍTÁS: Csatoljuk előre a görbületet (Feedforward), hogy be is kanyarodjon!
        u_applied = full(K_lqr * x_meas) + params.L * kappa_horizon(1);
    end
    
    % Biztonsági végleges korlátozások (Abszolút limit)
    u_applied = max(-0.5, min(0.5, u_applied));
    
    % Slew rate levágás (ha Fallback történt, ez biztosítja, hogy ne rántson nagyot)
    u_applied = max(u_prev_val - dU_max, min(u_prev_val + dU_max, u_applied));
    
    % Belső memória frissítése a következő híváshoz
    u_prev_val = u_applied;
end