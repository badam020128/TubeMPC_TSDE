%% ============================================================
%  MAIN SCRIPT
%  ------------------------------------------------------------
%  Project name   : Tube MPC optimization with neural network
%  Author         : Bene Ádám
%  Date           : 2026.04.28.
%  Description    : Main execution script that calls the required submodules and processing steps.
%  Note           : Run this file directly.
%% ============================================================
clear;
% reference path define for the controller
run("path_define.m")

% run tube MPC for previously defined reference path for data gathering
% purposes for further ML training
run("data_gather.m")

% TSDE (Two-Stage Deep Ensemble) with 5-5 neural networks for remaining
% error minimalization, and uncertainty prediction
run("TSDE.m")

% reference path define for the controller
run("path_define.m")


run("simulate.m")
