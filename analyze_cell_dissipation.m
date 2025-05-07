% Script to optimize statistical-thermodynamic cell model parameters and 
% analyze dissipative processes using Nelder-Mead optimization and Monte Carlo simulations.

clear all; close all; clc;

% Model parameters
n_params = 6;
Gr_true = 7.88700358722787;               % Pa, true value of the effective shear modulus of the cell in the local volume
Ge_true = 262193696.547693;               % Pa, true value of the effective shear modulus of the network of cytoskeletal segments in the local volume
lambda_true = 2.78937116692569;           % J/m^3, true effective volumetric energy of cross-links between filaments and microtubules in the local volume
gamma_true = 2.65017361763404e-21;        % m^3, true effective volume of a cytoskeletal segment
tau_0_true = 2.52994845476267e-08;        % s, true characteristic timescale of the cytoskeletal response
theta_true = 2.36e-22;                    % J, true effective temperature factor
true_params = [Gr_true, Ge_true, lambda_true, gamma_true, tau_0_true, theta_true];

% Load experimental data
load('cell_shear_exper_data.mat') % adopted from [W. J. Eldridge, A. Sheinfeld, M. T. Rinehart, and A. Wax, Imaging deformation of adherent cells due to shear stress using quantitative phase imaging, Opt. Lett. 41, 352 (2016).]
n_curves = length(sigma_vals); % Number of curves (4)
eps_true_cell = deformations(1, 1:n_curves); % Cell array with deformations
t_cell = times(1, 1:n_curves); % Cell array with times
eps_init = [eps_true_cell{1}(1) eps_true_cell{2}(1) eps_true_cell{3}(1) eps_true_cell{4}(1)];
eps_o_span = -0.49:0.01:0.98;

% Add noise to sigma (5% of maximum sigma)
noise_level = 0.05 * max(sigma_vals); % 0.0317 Pa
sigma_noisy = sigma_vals + normrnd(0, noise_level, size(sigma_vals));

% Define parameters to optimize
optimize_params = [true, true, true, true, true, false]; % If all parameters are false, then optimization based on syntactic data
fixed_values = true_params;
n_opt_params = sum(optimize_params); % Number of parameters to optimize

% Initialize variables for results
results = []; rmse_vals = []; median_relative_error = []; best_params = [];

if n_opt_params == 0
    % Generate synthetic data
    fprintf('Generating synthetic data with noise (noise level: %.1f%%)...\n', noise_level/max(sigma_vals)*100);
    eps_synthetic_cell = cell(1, n_curves);
    for i = 1:n_curves
        [~, eps_model] = ode15s(@(t, eps) deformation_rate(eps, Gr_true, Ge_true, lambda_true, ...
                                               gamma_true, tau_0_true, theta_true, ...
                                               sigma_noisy(i), eps_o_span), ...
                                t_cell{i}, eps_init(i), odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
        noise = noise_level * abs(eps_model) .* randn(size(eps_model));
        eps_synthetic_cell{i} = eps_model + noise;
    end

    % Visualize synthetic data
    figure(1);
    color = 'kbrm';
    for i = 1:n_curves
        [~, eps_model] = ode15s(@(t, eps) deformation_rate(eps, Gr_true, Ge_true, lambda_true, ...
                                               gamma_true, tau_0_true, theta_true, ...
                                               sigma_noisy(i), eps_o_span), ...
                                t_cell{i}, eps_init(i), odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
        hold on;
        plot(t_cell{i}, eps_synthetic_cell{i}, [color(i) '.'], 'LineWidth', 2, 'DisplayName', sprintf('σ = %.4f Pa', sigma_noisy(i)));
        plot(t_cell{i}, eps_model, [color(i) '-'], 'LineWidth', 2, 'DisplayName', 'Model');
        hold off;
        xlabel('t, s');

        ylabel('ε');
        legend('Location', 'east'); grid on;
    end
    pos = get(gcf,'Position'); set(gcf,'Position',[pos(1) pos(2) 1*pos(3) pos(4)]);

    % Compute RMSE and median relative error for synthetic data
    total_sum = 0; total_points = 0; rel_err = [];
    for i = 1:n_curves
        [~, eps_model] = ode15s(@(t, eps) deformation_rate(eps, Gr_true, Ge_true, lambda_true, ...
                                               gamma_true, tau_0_true, theta_true, ...
                                               sigma_noisy(i), eps_o_span), ...
                                t_cell{i}, eps_init(i), odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
        e_synth = eps_synthetic_cell{i};
        err = (e_synth - eps_model).^2;
        total_sum = total_sum + sum(err);
        total_points = total_points + length(err);
        rel_err = [rel_err; abs(e_synth - eps_model) ./ (abs(e_synth) + 1e-10) * 100];
    end
    rmse_synth = sqrt(total_sum / total_points);
    med_rel_err_synth = median(rel_err);
    fprintf('RMSE of synthetic data: %.4f\n', rmse_synth);
    fprintf('Median relative error: %.1f%%\n', med_rel_err_synth);

    % Analyze Nelder-Mead convergence
    fprintf('\nAnalyzing Nelder-Mead convergence on synthetic data...\n');
    N_initial = 100; % Number of runs
    n_phys = round(N_initial * 0.5); % 50% physically plausible
    n_rand = N_initial - n_phys; % 50% random
    initial_guesses = zeros(N_initial, n_params);
    rng('default');

    % Physically plausible initial guesses
    for i = 1:n_phys
        initial_guesses(i, :) = [...
            Gr_true * (1 + 0.1 * (2*rand-1)), ... % ±10% of true value
            Ge_true * (1 + 0.1 * (2*rand-1)), ...
            lambda_true * (1 + 0.1 * (2*rand-1)), ...
            gamma_true * (1 + 0.1 * (2*rand-1)), ...
            tau_0_true * (1 + 0.1 * (2*rand-1)), ...
            fixed_values(6)];
    end
    % Random initial guesses
    for i = n_phys+1:N_initial
        initial_guesses(i, :) = [...
            1e0 + (1e3-1e0)*rand, ...
            1e6 + (1e9-1e6)*rand, ...
            1e0 + (1e1-1e0)*rand, ...
            1e-20 + (1e-22-1e-20)*rand, ...
            1e-8 + (1e-11-1e-8)*rand, ...
            fixed_values(6)];
    end

    % Normalize initial guesses
    scale = 10.^(floor(log10(initial_guesses(1, :)))); % Scales for normalization
    opt_indices = [1, 2, 3, 4, 5]; % Optimize G_r, G_e, lambda, gamma, tau_0
    n_opt_params = length(opt_indices);
    initial_guesses_norm = initial_guesses(:, opt_indices) ./ scale(opt_indices);

    % Optimization
    options = optimset('Display', 'off', 'MaxIter', 1000, 'TolFun', 1e-6, 'TolX', 1e-6);
    results_norm = zeros(N_initial, n_opt_params);
    results = zeros(N_initial, n_params);
    rmse_vals = zeros(N_initial, 1);
    median_relative_error = zeros(N_initial, 1);
    tic;
    parfor i = 1:N_initial
        [params_norm, ~] = fminsearch(@(x) rmse(x, t_cell, eps_synthetic_cell, sigma_noisy, eps_init, ...
                                                eps_o_span, scale, [true, true, true, true, true, false], ...
                                                fixed_values, opt_indices), ...
                                      initial_guesses_norm(i, :), options);
        results_norm(i, :) = params_norm;
        params_full = fixed_values;
        params_full(opt_indices) = params_norm .* scale(opt_indices);
        results(i, :) = params_full;
        [rmse_val, med_rel_err] = rmse(params_norm, t_cell, eps_synthetic_cell, sigma_noisy, eps_init, ...
                                       eps_o_span, scale, [true, true, true, true, true, false], ...
                                       fixed_values, opt_indices);
        rmse_vals(i) = rmse_val;
        median_relative_error(i) = med_rel_err;
    end
    toc;

    % Visualize RMSE convergence
    figure(2);
    plot(1:n_phys, median_relative_error(1:n_phys), 'bo-', 'LineWidth', 2, 'DisplayName', 'Physical initials');
    hold on;
    plot(n_phys+1:N_initial, median_relative_error(n_phys+1:end), 'r*-', 'LineWidth', 2, 'DisplayName', 'Random initials');
    xlabel('Run index'); ylabel('Relative error, %');
    legend('Location', 'best'); grid on;
    pos = get(gcf,'Position'); set(gcf,'Position',[pos(1) pos(2) 2*pos(3) pos(4)]);

    % Prepare data for output
    [min_rmse, best_idx] = min(rmse_vals);
    best_params = results(best_idx, :);
    data_cell = eps_synthetic_cell;
    is_synthetic = true;
else
    % Optimization on real data
    N_initial = 100;
    initial_guesses = zeros(N_initial, n_params);
    rng('default');
    for i = 1:N_initial
        for j = 1:n_params
            if ~optimize_params(j)
                initial_guesses(i, j) = fixed_values(j);
            else
                switch j
                    case 1
                        initial_guesses(i, j) = 1e0 + (1e3-1e0)*rand;
                    case 2
                        initial_guesses(i, j) = 1e6 + (1e9-1e6)*rand;
                    case 3
                        initial_guesses(i, j) = 1e0 + (1e1-1e0)*rand;
                    case 4
                        initial_guesses(i, j) = 1e-20 + (1e-22-1e-20)*rand;
                    case 5
                        initial_guesses(i, j) = 1e-8 + (1e-11-1e-8)*rand;
                    case 6
                        initial_guesses(i, j) = (300 + (400-300)) * 1.38e-23 * rand;
                end
            end
        end
    end
            
    scale = 10.^(floor(log10(initial_guesses(1, :)))); % Scales for normalization
    initial_guesses_norm = zeros(N_initial, n_opt_params);
    opt_indices = find(optimize_params);
    for i = 1:N_initial
        initial_guesses_norm(i, :) = initial_guesses(i, opt_indices) ./ scale(opt_indices);
    end

    options = optimset('Display', 'off', 'MaxIter', 1000, 'TolFun', 1e-6, 'TolX', 1e-6);
    results_norm = zeros(N_initial, n_opt_params);
    results = zeros(N_initial, n_params);
    rmse_vals = zeros(N_initial, 1);
    median_relative_error = zeros(N_initial, 1);
    tic;
    parfor i = 1:N_initial
        [params_norm, ~] = fminsearch(@(x) rmse(x, t_cell, eps_true_cell, sigma_vals, eps_init, ...
                                                eps_o_span, scale, optimize_params, fixed_values, opt_indices), ...
                                      initial_guesses_norm(i, :), options);
        results_norm(i, :) = params_norm;
        params_full = fixed_values;
        params_full(opt_indices) = params_norm .* scale(opt_indices);
        results(i, :) = params_full;
        [rmse_val, med_rel_err] = rmse(params_norm, t_cell, eps_true_cell, sigma_vals, eps_init, ...
                                       eps_o_span, scale, optimize_params, fixed_values, opt_indices);
        rmse_vals(i) = rmse_val;
        median_relative_error(i) = med_rel_err;
    end
    toc;

    [min_rmse, best_idx] = min(rmse_vals);
    best_params = results(best_idx, :);
    data_cell = eps_true_cell;
    is_synthetic = false;

    figure(3);
    plot(1:N_initial, median_relative_error, 'k*-', 'LineWidth', 2, 'DisplayName', 'Random initials');
    xlabel('Run index'); ylabel('Relative error, %');
    legend('Location', 'best'); grid on;
    pos = get(gcf,'Position'); set(gcf,'Position',[pos(1) pos(2) 2*pos(3) pos(4)]);
end
%%
% Display results
fprintf('\nOptimal parameters (RMSE = %.4f, Median error = %.1f%%):\n', ...
        min_rmse, median_relative_error(best_idx));
if is_synthetic
    fprintf('G_r: %.0f Pa (true: %.0f Pa, deviation: %.1f%%)\n', ...
            best_params(1), Gr_true, abs(best_params(1)-Gr_true)/Gr_true*100);
    fprintf('G_e: %.1f MPa (true: %.1f MPa, deviation: %.1f%%)\n', ...
            best_params(2)/1e6, Ge_true/1e6, abs(best_params(2)-Ge_true)/Ge_true*100);
    fprintf('lambda: %.2e J/m^3 (true: %.2e J/m^3, deviation: %.1f%%)\n', ...
            best_params(3), lambda_true, abs(best_params(3)-lambda_true)/lambda_true*100);
    fprintf('gamma: %.2e m^3 (true: %.2e m^3, deviation: %.1f%%)\n', ...
            best_params(4), gamma_true, abs(best_params(4)-gamma_true)/gamma_true*100);
    fprintf('tau_0: %.2e s (true: %.2e s, deviation: %.1f%%)\n', ...
            best_params(5), tau_0_true, abs(best_params(5)-tau_0_true)/tau_0_true*100);
else
    fprintf('G_r: %.0f Pa\n', best_params(1));
    fprintf('G_e: %.1f MPa\n', best_params(2)/1e6);
    fprintf('lambda: %.2e J/m^3\n', best_params(3));
    fprintf('gamma: %.2e m^3\n', best_params(4));
    fprintf('tau_0: %.2e s\n', best_params(5));
end
fprintf('theta: %.2e J\n', best_params(6));

threshold = 10;

fprintf('\nStatistics across runs:\n');
r = results(median_relative_error < threshold,:);
group = repmat(1:6, size(r,1), 1);
group = group(:);
results_reshaped = r(:);
[mu, sigma] = grpstats(results_reshaped, group, {'mean', 'meanci'});
fprintf('G_r: %.0f ± %.0f Pa\n', mu(1), sigma(1));
fprintf('G_e: %.1f ± %.1f MPa\n', mu(2)/1e6, sigma(2)/1e6);
fprintf('lambda: %.2e ± %.2e J/m^3\n', mu(3), sigma(3));
fprintf('gamma: %.2e ± %.2e m^3\n', mu(4), sigma(4));
fprintf('tau_0: %.2e ± %.2e s\n', mu(5), sigma(5));
fprintf('theta: %.2e J\n', mu(6));
fprintf('Mean RMSE: %.4f\n', mean(rmse_vals));
fprintf('Median error: %.1f%%\n', mean(median_relative_error(median_relative_error < threshold)));

success_rate = sum(median_relative_error < threshold) / size(results, 1) * 100;
fprintf('Fraction of runs with relative error < %i%%: %.1f%%\n', threshold, success_rate);

% Correlation analysis
corr_matrix = corr(results);
param_names = {'G_r', 'G_e', 'lambda', 'gamma', 'tau_0', 'theta'};
fprintf('\nPearson correlation matrix:\n');
fprintf('         G_r      G_e      lambda   gamma    tau_0    theta\n');
for i = 1:n_params
    fprintf('%s:   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f\n', param_names{i}, corr_matrix(i, :));
end

fprintf('\nCorrelation interpretation:\n');
for i = 1:n_params
    for j = i+1:n_params
        r = corr_matrix(i, j);
        if abs(r) > 0.7
            fprintf('Strong correlation between %s and %s (r = %.3f): possible non-uniqueness.\n', ...
                    param_names{i}, param_names{j}, r);
        elseif abs(r) > 0.3
            fprintf('Moderate correlation between %s and %s (r = %.3f): weak influence.\n', ...
                    param_names{i}, param_names{j}, r);
        else
            fprintf('Weak correlation between %s and %s (r = %.3f): parameters independent.\n', ...
                    param_names{i}, param_names{j}, r);
        end
    end
end

% Visualize curves
figure(4);
color = 'kbrm';
for i = 1:n_curves
    if is_synthetic
        sigma_tmp = sigma_noisy(i);
    else
        sigma_tmp = sigma_vals(i);
    end
    [~, eps_model] = ode15s(@(t, eps) deformation_rate(eps, best_params(1), best_params(2), best_params(3), ...
                                            best_params(4), best_params(5), best_params(6), ...
                                            sigma_tmp, eps_o_span), ...
                            t_cell{i}, eps_init(i), odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
    hold on;
    plot(t_cell{i}, data_cell{i}, [color(i) '.'], 'LineWidth', 2, 'DisplayName', sprintf('σ = %.4f Pa', sigma_tmp));
    plot(t_cell{i}, eps_model, [color(i) '-'], 'LineWidth', 2, 'DisplayName', 'Model');
    hold off;
    xlabel('t, s'); ylabel('ε');
    legend('Location', 'best','NumColumns',2); grid on;
end
pos = get(gcf,'Position'); set(gcf,'Position',[pos(1) pos(2) 1*pos(3) pos(4)]);

% Plot histograms
figure(5);
subplot(2, 3, 1); histogram(results(:, 1), 20, 'FaceAlpha', 0.5);
xlabel('G_r, Pa'); ylabel('Frequency');
subplot(2, 3, 2); histogram(results(:, 2)/1e6, 20, 'FaceAlpha', 0.5);
xlabel('G_e, MPa'); ylabel('Frequency');
subplot(2, 3, 3); histogram(results(:, 3), 20, 'FaceAlpha', 0.5);
xlabel('lambda, J/m^3'); ylabel('Frequency');
subplot(2, 3, 4); histogram(results(:, 4), 20, 'FaceAlpha', 0.5);
xlabel('gamma, m^3'); ylabel('Frequency');
subplot(2, 3, 5); histogram(results(:, 5), 20, 'FaceAlpha', 0.5);
xlabel('tau_0, s'); ylabel('Frequency');
subplot(2, 3, 6); histogram(results(:, 6), 20, 'FaceAlpha', 0.5);
xlabel('theta, J'); ylabel('Frequency');
pos = get(gcf,'Position'); set(gcf,'Position',[pos(1) pos(2) 2*pos(3) pos(4)]);

% Plot correlation matrix
figure(6);
imagesc(corr_matrix); colorbar;
set(gca, 'XTick', 1:n_params, 'XTickLabel', param_names, 'YTick', 1:n_params, 'YTickLabel', param_names);
for i = 1:n_params
    for j = 1:n_params
        text(j, i, sprintf('%.2f', corr_matrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'w');
    end
end
c = colorbar;
c.Label.String = 'r';
colormap('jet');

% Compute dissipative function
[product_vals, eps_o_vals] = calculate_dissipative_function(best_params, sigma_vals, sigma_noisy, is_synthetic, ...
                                              t_cell, eps_init, eps_o_span);

% Compute entropy
compute_entropy(product_vals, t_cell, sigma_vals);

% Calculate normalized dissipative energy
diss_energy_norm = zeros(1, n_curves);
diss_energy = zeros(1, n_curves);
G_r = best_params(1);
V = 1; T = 310;
threshold = 0.1;
for i = 1:n_curves
    eps_o_squared = mean(eps_o_vals{i}.^2);
    prod = product_vals{i} / T;
    max_prod = max(prod);
    idx = find(prod < threshold * max_prod, 1, 'first');
    if ~isempty(idx)
        tau = t_cell{i}(idx);
    else
        tau = t_cell{i}(end);
    end
    diss_energy(i) = trapz(t_cell{i}, product_vals{i});
    diss_energy_norm(i) = diss_energy(i) / (G_r * eps_o_squared * tau * V);
end
fprintf('\nCombined normalization (G_r * eps_o^2 * tau * V):\n');
for i = 1:n_curves
    fprintf('Curve %d (σ = %.4f Pa): E_diss_norm_combined = %.4e\n', ...
            i, sigma_vals(i), diss_energy_norm(i));
end
fprintf('Mean E_diss_norm_combined: %.4e ± %.4e\n', ...
        mean(diss_energy_norm), std(diss_energy_norm));
fprintf('Relative standard deviation of E_diss_norm_combined: %.4e\n', ...
        std(diss_energy_norm) / mean(diss_energy_norm));

% Monte Carlo Dissipation Analysis
monte_carlo_dissipation_analysis



%% Function's section
% Objective function for optimization (RMSE)
function [error, med_rel_err] = rmse(parameters_norm, t_cell, e_exp_cell, sigma_vals, eps_init, ...
                                     eps_o_span, scale, optimize_params, fixed_values, opt_indices)
    parameters = fixed_values;
    parameters(opt_indices) = parameters_norm .* scale(opt_indices);
    Gr = parameters(1); Ge = parameters(2); lambda = parameters(3);
    gamma = parameters(4); tau_0 = parameters(5); theta = parameters(6);

    rel_err = []; total_sum = 0; total_points = 0;
    for i = 1:length(sigma_vals)
        try
            [~, eps_model] = ode15s(@(t, eps) deformation_rate(eps, Gr, Ge, lambda, gamma, tau_0, theta, ...
                                                    sigma_vals(i), eps_o_span), ...
                                    t_cell{i}, eps_init(i), odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
            e_exp = medfilt1(e_exp_cell{i});
            err = (e_exp - eps_model).^2;
            total_sum = total_sum + sum(err);
            total_points = total_points + length(err);
            rel_err = [rel_err; abs(e_exp - eps_model) ./ (abs(e_exp) + 1e-10) * 100];
        catch
            warning('Error in ode15s for sigma = %.4f Pa', sigma_vals(i));
            total_sum = total_sum + 1e6;
            total_points = total_points + length(t_cell{i});
            rel_err = [rel_err; 1e6 * ones(length(t_cell{i}), 1)];
        end
    end

    error = sqrt(total_sum / total_points);
    rel_err(isinf(rel_err) | isnan(rel_err)) = 1e6;
    med_rel_err = median(rel_err);
    if isinf(error) || isnan(error)
        error = 1e6;
    end
end

% Compute total deformation as a function of time under constant stress
function deformation = compute_deformation(t, G_r, G_e, lambda, gamma, tau_0, theta, sigma, eps_init, eps_range)
    % Inputs:
    %   t         - Time array
    %   G_r       - Shear modulus of the local cell volume (Pa)
    %   G_e       - Shear modulus of network of cytoskeletal segments (Pa)
    %   lambda    - Crosslink energy (J/m^3)
    %   gamma     - Segment volume (m^3)
    %   tau_0     - Relaxation time (s)
    %   theta     - Effective temperature factor (J)
    %   sigma     - Applied stress (Pa)
    %   eps_init  - Initial deformation
    %   eps_range - Deformation range for calculations
    % Output:
    %   deformation - Computed deformation over time

    [~, deformation] = ode15s(@(t, eps) deformation_rate(eps, G_r, G_e, lambda, gamma, tau_0, theta, sigma, eps_range), ...
                              t, eps_init, odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
end

% Compute deformation rate for solving differential equation
function dedt = deformation_rate(eps, G_r, G_e, lambda, gamma, tau_0, theta, sigma, eps_range)
    % Inputs:
    %   eps       - Current deformation
    %   G_r       - Shear modulus (Pa)
    %   G_e       - Shear modulus of network of cytoskeletal segments (Pa)
    %   lambda    - Crosslink energy (J/m^3)
    %   gamma     - Segment volume (m^3)
    %   tau_0     - Relaxation time (s)
    %   theta     - Effective temperature factor (J)
    %   sigma     - Applied stress (Pa)
    %   eps_range - Deformation range for calculations
    % Output:
    %   dedt      - Deformation rate

    % Calculate auxiliary variables
    chi = theta / (lambda * gamma);
    eps_0 = eps - ((sigma - G_r * eps) / G_e);
    sigma_0 = sigma - eps * G_r;
    
    % Compute ksi and viscosity
    ksi_val = compute_ksi(eps_0);
    nu = compute_viscosity(sigma_0 * (gamma / theta), eps_range, chi, G_e, tau_0);
    
    % Compute deformation rate
    dedt = (G_e / (nu * (G_e + G_r))) * (sigma - G_r * eps - (theta / gamma) * ksi_val + lambda * eps_0);
end

% Compute ksi value based on deformation
function ksi = compute_ksi(eps_0)
    % Input:
    %   eps_0 - Reference deformation
    % Output:
    %   ksi   - Computed ksi value

    numerator = 260 + eps_0 .* (123200 + (76250 - 114200 * eps_0) .* eps_0);
    denominator = 12400 + eps_0 .* (16770 + eps_0 .* (-20670 + (-8820 + eps_0) .* eps_0));
    ksi = numerator ./ denominator;
end

% Compute potential function Psi
function psi = compute_potential(s, eps, chi)
    % Inputs:
    %   s   - Normalized stress
    %   eps - Deformation
    %   chi - Material parameter
    % Output:
    %   psi - Computed potential

    psi = real(-1.7 * log(0.99 - eps) - 114161 * log(8822.34 - eps) - ...
               0.9 * log(0.5 + eps) - 36.34 * log(2.84 + eps) - ...
               (eps.^2) / (2 * chi) - s .* eps);
    
    % Normalize potential by subtracting mean
    means = mean(psi, 2);
    psi = psi - means;
end

% Compute potential barrier height
function delta_psi = compute_potential_barrier(s, eps, chi)
    % Inputs:
    %   s   - Normalized stress
    %   eps - Deformation
    %   chi - Material parameter
    % Output:
    %   delta_psi - Potential barrier height

    % Compute potential
    psi = compute_potential(s, eps, chi);
    
    % Identify local maxima and minima
    is_max = islocalmax(psi);
    is_min = islocalmin(psi);
    
    % Handle cases based on presence of extrema
    if numel(eps(is_max)) + numel(eps(is_min)) == 0
        if psi(1) > psi(end)
            delta_psi = 0;
        else
            delta_psi = psi(end) - psi(1);
        end
    else
        if numel(eps(is_max)) + numel(eps(is_min)) > 1
            max_idx = find(is_max, 1);
            min_idx = find(is_min, 1);
            if eps(max_idx) > eps(min_idx)
                delta_psi = psi(max_idx) - psi(min_idx);
            else
                error('Error: Maximum deformation less than minimum deformation');
            end
        else
            if psi(1) > psi(end)
                delta_psi = 0;
            else
                delta_psi = psi(end) - psi(is_min);
            end
        end
    end
end

% Compute orientational viscosity
function nu = compute_viscosity(s_0d, eps, chi, G_e, tau_0)
    % Inputs:
    %   s_0d  - Normalized stress
    %   eps   - Deformation
    %   chi   - Material parameter
    %   G_e   - Shear modulus of network of cytoskeletal segments (Pa)
    %   tau_0 - Relaxation time (s)
    % Output:
    %   nu    - Orientational viscosity

    nu = G_e * tau_0 * exp(compute_potential_barrier(s_0d, eps, chi));
end

% Compute dissipative function
function [product_vals, eps_o_vals] = calculate_dissipative_function(best_params, sigma_vals, sigma_noisy, is_synthetic, ...
                                                      t_cell, eps_init, eps_o_span)
    % Compute the product d(eps_o)/dt * (sigma_o - dF/d(eps_o))
    fprintf('\nComputing the product d(eps_o)/dt * (sigma_o - dF/d(eps_o))...\n');

    % Parameters for calculations
    chi = best_params(6) / (best_params(3) * best_params(4)); % chi = theta / (lambda * gamma)
    n_curves = length(sigma_vals);
    product_vals = cell(1, n_curves); % Store products
    eps_o_vals = cell(1, n_curves); % Store eps_o

    % Loop over curves
    for i = 1:n_curves
        if is_synthetic
            sigma_tmp = sigma_noisy(i);
        else
            sigma_tmp = sigma_vals(i);
        end
        
        % Solve ODE to obtain eps
        [t, eps_model] = ode15s(@(t, eps) deformation_rate(eps, best_params(1), best_params(2), best_params(3), ...
                                                best_params(4), best_params(5), best_params(6), ...
                                                sigma_tmp, eps_o_span), ...
                                t_cell{i}, eps_init(i), odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
        
        % Compute eps_o, sigma_o, and d(eps_o)/dt
        eps_o = eps_model - (sigma_tmp - best_params(1) * eps_model) / best_params(2);
        sigma_o = sigma_tmp - best_params(1) * eps_model;
        dedt = deformation_rate(eps_model, best_params(1), best_params(2), best_params(3), ...
                    best_params(4), best_params(5), best_params(6), sigma_tmp, eps_o_span);
        
        % Numerically compute dF/d(eps_o)
        dF_deps_o = zeros(size(eps_o));
        h = 1e-6; % Step for numerical derivative
        for j = 1:length(eps_o)
            % Potential F for eps_o and eps_o + h
            F = compute_potential(sigma_o(j) * (best_params(4)/best_params(6)), eps_o(j), chi);
            F_h = compute_potential(sigma_o(j) * (best_params(4)/best_params(6)), eps_o(j) + h, chi);
            dF_deps_o(j) = (F_h - F) / h; % Numerical derivative
        end
        
        % Compute product
        product_vals{i} = dedt .* (sigma_o - dF_deps_o);
        eps_o_vals{i} = eps_o;
    end

    % Visualize product dependence on eps_o
    color = 'kbrm';
    for i = 1:n_curves
        figure(7);
        hold on;
        yyaxis left;
        semilogy(eps_o_vals{i}, product_vals{i}, [color(i) '-'], 'LineWidth', 2, ...
             'DisplayName', sprintf('σ = %.4f Pa', sigma_vals(i)));
        ylabel('P_s,-');
        yyaxis right;
        plot(eps_o_vals{i}, cumtrapz(t_cell{i}, product_vals{i} / 310), ...
          [color(i) '--'], 'LineWidth', 2, ...
             'DisplayName', sprintf('σ = %.4f Pa', sigma_vals(i)));
        ylabel('S,--');
        hold off;
        xlabel('ε_o');
        legend('Location', 'best', 'NumColumns', 2); grid on;

        figure(8);
        hold on;
        yyaxis left;
        plot(t_cell{i}, product_vals{i}, [color(i) '-'], 'LineWidth', 2, ...
             'DisplayName', sprintf('σ = %.4f Pa', sigma_vals(i)));
        ylabel('P_s,-');
        yyaxis right;
        plot(t_cell{i}, cumtrapz(t_cell{i}, product_vals{i} / 310), ...
          [color(i) '--'], 'LineWidth', 2, ...
             'DisplayName', sprintf('σ = %.4f Pa', sigma_vals(i)));
        ylabel('S,--');
        hold off;
        xlabel('t, s');
        legend('Location', 'best', 'NumColumns', 2); grid on;
    end
    pos = get(gcf, 'Position'); set(gcf, 'Position', [pos(1) pos(2) 2*pos(3) pos(4)]);

    % Product statistics
    fprintf('\nProduct statistics:\n');
    for i = 1:n_curves
        mean_product = mean(product_vals{i});
        std_product = std(product_vals{i});
        fprintf('Curve %d (σ = %.4f Pa): Mean = %.2e, Std. dev. = %.2e\n', ...
                i, sigma_vals(i), mean_product, std_product);
    end
end

% Compute entropy
function compute_entropy(product_vals, t_cell, sigma_vals)
    % Normalize and compute area
    T = 310; % Temperature in Kelvin
    n_curves = length(sigma_vals);
    entropy_vals = cell(1, n_curves);
    areas = zeros(1, n_curves);
    for i = 1:n_curves
        % Compute S(t)
        entropy_vals{i} = cumtrapz(t_cell{i}, product_vals{i} / T);
        % Normalize by sigma
        entropy_norm = entropy_vals{i} / sigma_vals(i);
        % Compute area
        areas(i) = trapz(t_cell{i}, entropy_norm);
    end

    % Display results
    fprintf('\nArea under normalized entropy curves:\n');
    for i = 1:n_curves
        fprintf('Curve %d (σ = %.4f Pa): Area = %.4e\n', i, sigma_vals(i), areas(i));
    end

    % Check consistency
    fprintf('Mean area: %.4e\n', mean(areas));
    fprintf('Std. deviation of area: %.4e\n', std(areas));
end

% Monte Carlo dissipation analysis
function monte_carlo_dissipation_analysis()
    % Load optimal model parameters
    best_params = [7.88700358722787, 262193696.547693, 2.78937116692569, ...
                   2.65017361763404e-21, 2.52994845476267e-08, 2.36e-22];

    % Monte Carlo parameters
    n_samples = 10000; % Number of random samples
    variation_percent = 10; % Parameter variation percentage

    % Load experimental data
    data = load('cell_shear_exper_data.mat');
    sigma_vals = data.sigma_vals;
    times = data.times;
    deformations = data.deformations;
    n_curves = length(sigma_vals);
    eps_init = [deformations{1}(1), deformations{2}(1), deformations{3}(1), deformations{4}(1)];
    eps_o_span = -0.49:0.01:0.98;

    % Generate random parameters
    rng('default'); % For reproducibility
    params_samples = zeros(n_samples, length(best_params));

    for i = 1:length(best_params)
        % Vary each parameter within ±variation_percent%
        lower_bound = best_params(i) * (1 - variation_percent/100);
        upper_bound = best_params(i) * (1 + variation_percent/100);
        params_samples(:,i) = unifrnd(lower_bound, upper_bound, [n_samples, 1]);
    end
    params_samples(:,end) = 2.36e-22;

    % Analyze dissipative energy for each sample
    diss_energy_norm = zeros(n_samples, n_curves);

    parfor k = 1:n_samples
        current_params = params_samples(k,:);

        % Compute dissipative function for current parameters
        [product_vals, eps_o_vals] = calculate_dissipative_function_mc(...
            current_params, sigma_vals, deformations, times, eps_init, eps_o_span);

        % Calculate normalized dissipative energy
        for i = 1:n_curves
            eps_o_squared = mean(eps_o_vals{i}.^2);
            G_r = current_params(1);
            V = 1; T = 310; % Volume and temperature
            threshold = 0.1;

            prod = product_vals{i} / T;
            max_prod = max(prod);
            idx = find(prod < threshold * max_prod, 1, 'first');

            if ~isempty(idx)
                tau = times{i}(idx);
            else
                tau = times{i}(end);
            end

            diss_energy = trapz(times{i}, product_vals{i});
            diss_energy_norm(k,i) = diss_energy / (G_r * eps_o_squared * tau * V);
        end
    end

    % Analyze results
    analyze_results(params_samples, diss_energy_norm, sigma_vals, best_params);
end

% Simplified dissipative function for Monte Carlo analysis
function [product_vals, eps_o_vals] = calculate_dissipative_function_mc(params, sigma_vals, deformations, times, eps_init, eps_o_span)
    chi = params(6) / (params(3) * params(4));
    n_curves = length(sigma_vals);
    product_vals = cell(1, n_curves);
    eps_o_vals = cell(1, n_curves);
    
    for i = 1:n_curves
        [~, eps_model] = ode15s(@(t, eps) deformation_rate(eps, params(1), params(2), params(3), ...
                              params(4), params(5), params(6), sigma_vals(i), eps_o_span), ...
                          times{i}, eps_init(i), odeset('RelTol', 1e-6));
        
        eps_o = eps_model - (sigma_vals(i) - params(1) * eps_model) / params(2);
        sigma_o = sigma_vals(i) - params(1) * eps_model;
        dedt = deformation_rate(eps_model, params(1), params(2), params(3), ...
               params(4), params(5), params(6), sigma_vals(i), eps_o_span);
        
        % Simplified derivative computation
        h = 1e-6;
        dF_deps_o = (compute_potential(sigma_o * (params(4)/params(6)), eps_o + h, chi) - ...
                   compute_potential(sigma_o * (params(4)/params(6)), eps_o, chi)) / h;
        
        product_vals{i} = dedt .* (sigma_o - dF_deps_o);
        eps_o_vals{i} = eps_o;
    end
end

% Analyze and visualize Monte Carlo results
function analyze_results(params_samples, diss_energy_norm, sigma_vals, best_params)
    % Statistics of normalized dissipative energy
    mean_diss = mean(diss_energy_norm, 1);
    std_diss = std(diss_energy_norm, 0, 1);
    cv_diss = std_diss ./ mean_diss * 100; % Coefficient of variation
    
    fprintf('\nStatistics of normalized dissipative energy:\n');
    for i = 1:length(sigma_vals)
        fprintf('Curve %d (σ = %.4f Pa): Mean = %.4e, Std. dev. = %.4e, CV = %.1f%%\n', ...
                i, sigma_vals(i), mean_diss(i), std_diss(i), cv_diss(i));
    end
    fprintf('Mean = %.4e, Std. dev. = %.4e, CV = %.1f%%\n', ...
                mean(mean_diss), mean(std_diss), mean(cv_diss));
    
    % Dependence of dissipation on parameters
    param_names = {'G_r', 'G_e', 'lambda', 'gamma', 'tau_0', 'theta'};
    figure('Position', [100, 100, 1200, 800]);
    
    for i = 1:6
        subplot(2,3,i);
        scatter(params_samples(:,i), mean(diss_energy_norm, 2), 'filled');
        xlabel(param_names{i}); ylabel('Normalized dissipative energy');
        grid on;
    end
    
    % Correlation of parameters with dissipation
    corr_matrix = corr([params_samples, mean(diss_energy_norm, 2)]);
    fprintf('\nCorrelation of parameters with dissipation:\n');
    for i = 1:6
        fprintf('%s: r = %.3f\n', param_names{i}, corr_matrix(i,end));
    end
    
    % Distribution of dissipative energy
    figure(9);
    histogram(mean(diss_energy_norm, 2), 30, 'FaceAlpha', 0.7);
    xlabel('Mean normalized dissipative energy'); ylabel('Frequency');
    grid on;
    
    % Comparison with optimal parameters
    optimal_diss = diss_energy_norm(1,:); % First sample - optimal parameters
    figure(10);
    boxplot(diss_energy_norm);
    hold on;
    plot(1:size(diss_energy_norm,2), optimal_diss, 'ro-', 'LineWidth', 2);
    xlabel('Curve'); ylabel('Normalized dissipative energy');
    grid on;
end