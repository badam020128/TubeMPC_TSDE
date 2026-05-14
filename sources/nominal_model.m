function [A, B, G] = nominal_model(v, L, dt)
    % NOMINÁLIS LINEÁRIS MODELL GENERÁLÓ
    % ------------------------------------------------------------
    % Bemenetek:
    %   v  : hosszirányú sebesség [m/s]
    %   L  : tengelytáv [m]
    %   dt : mintavételi idő [s]
    %
    % Kimenetek:
    %   A, B : Diszkrét idejű állapotteres mátrixok (x_k1 = A*x_k + B*u_k)
    %   G    : Görbületi (kappa) hatást leíró vektor
    % ------------------------------------------------------------

    % Állapotok: x = [e_y; e_psi] (laterális hiba, irányszöghiba)
    % Bemenet:  u = delta (kormányszög)
    
    % Rendszermátrix (A): leírja, hogyan változik a hiba beavatkozás nélkül
    % e_y(k+1)   = e_y(k) + v * dt * e_psi(k)
    % e_psi(k+1) = e_psi(k)
    A = [1,  v*dt;
         0,  1];

    % Beavatkozó mátrix (B): leírja a kormányzás hatását
    % e_y(k+1) nem függ közvetlenül a kormányzástól (csak az e_psi-n keresztül)
    % e_psi(k+1) = e_psi(k) + (v/L) * dt * delta
    B = [0;
         (v/L)*dt];

    % Zavarást/Pályát leíró vektor (G): a pálya görbületének hatása
    % e_psi_next = ... - v * kappa * dt
    G = [0; 
        -v*dt];
end