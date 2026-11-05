% =========================================================================
% SCRIPT: AI-Driven Deep Learning for Predicting Copper Adsorption on 
%         Zeolite-Silver Nanocomposites
% 
% DESCRIPTION:
%   Computational script for the preprint (SSRN DOI: 10.2139/ssrn.5194712).
%   Evaluates 8 Machine Learning algorithms and generates Figures 3, 4 & 5.
%
% REQUIREMENTS:
%   - MATLAB R2024a (or newer)
%   - Deep Learning Toolbox, Statistics and Machine Learning Toolbox
% =========================================================================

clear; clc; close all; warning off;

%% 1. CARGA Y PREPARACIÓN DE DATOS EXPERIMENTALES

% Datos experimentales Zeolite (29 puntos, 0-840 min)
t_Zeolite = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, ...
             390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720, ...
             750, 780, 810, 840]';
perc_Zeolite = [0, 15.68, 24.87, 35.2, 41.8, 46.7, 51.5, 56.3, 61.1, 64.2, ...
                66.7, 69.1, 71.3, 72.5, 75.1, 77.7, 79.8, 81.7, 83.4, 84.7, ...
                86.1, 87.2, 87.8, 90.1, 90.8, 90.6, 91.1, 91.3, 91.7]';
qt_Zeolite = [0, 0.003136, 0.004974, 0.00704, 0.00836, 0.00934, 0.0103, ...
              0.01126, 0.01222, 0.01284, 0.01334, 0.01382, 0.01426, 0.0145, ...
              0.01502, 0.01554, 0.01596, 0.01634, 0.01668, 0.01694, 0.01722, ...
              0.01744, 0.01756, 0.01802, 0.01816, 0.01812, 0.01822, 0.01826, ...
              0.01834]';

% Datos experimentales Zeolite-Ag (17 puntos, 0-480 min)
t_Zeolite_Ag = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, ...
                360, 390, 420, 450, 480]';
perc_Zeolite_Ag = [0, 13, 27.3, 34.2, 44.4, 53.3, 55.9, 64.6, 78.8, 79.2, ...
                   84.7, 90, 92.1, 94.5, 97.3, 98.1, 97.8]';
qt_Zeolite_Ag = [0, 0.0026, 0.00546, 0.00684, 0.00888, 0.01066, 0.01118, ...
                 0.01292, 0.01576, 0.01584, 0.01694, 0.018, 0.01842, ...
                 0.0189, 0.01946, 0.01962, 0.01956]';

C0 = 2.5; % mg/L

%% 2. DENSIFICACIÓN CON MODELO LOGÍSTICO

t_dense_Zeolite = (0:2:840)';
t_dense_Zeolite_Ag = (0:2:480)';

ft = fittype('Rmax*(t^n)/(t^n + t50^n)', 'independent', 't', 'coefficients', {'Rmax', 't50', 'n'});

fit_Zeolite = fit(t_Zeolite, perc_Zeolite, ft, 'StartPoint', [92, 300, 3], 'Lower', [85, 50, 1], 'Upper', [100, 1000, 10], 'Robust', 'Bisquare');
perc_dense_Zeolite = fit_Zeolite(t_dense_Zeolite);
qt_dense_Zeolite = interp1(t_Zeolite, qt_Zeolite, t_dense_Zeolite, 'spline');

fit_Zeolite_Ag = fit(t_Zeolite_Ag, perc_Zeolite_Ag, ft, 'StartPoint', [98, 200, 4], 'Lower', [90, 50, 1], 'Upper', [100, 500, 10], 'Robust', 'Bisquare');
perc_dense_Zeolite_Ag = fit_Zeolite_Ag(t_dense_Zeolite_Ag);
qt_dense_Zeolite_Ag = interp1(t_Zeolite_Ag, qt_Zeolite_Ag, t_dense_Zeolite_Ag, 'spline');

%% 3. PREPARACIÓN DE DATOS PARA MACHINE LEARNING

X_Zeolite = [t_dense_Zeolite, qt_dense_Zeolite, ones(size(t_dense_Zeolite))*C0];
Y_Zeolite = perc_dense_Zeolite;
X_Zeolite_Ag = [t_dense_Zeolite_Ag, qt_dense_Zeolite_Ag, ones(size(t_dense_Zeolite_Ag))*C0];
Y_Zeolite_Ag = perc_dense_Zeolite_Ag;

[X_norm_Zeolite, PSx_Zeolite] = mapminmax(X_Zeolite', 0, 1);
[Y_norm_Zeolite, PSy_Zeolite] = mapminmax(Y_Zeolite', 0, 1);
[X_norm_Zeolite_Ag, PSx_Zeolite_Ag] = mapminmax(X_Zeolite_Ag', 0, 1);
[Y_norm_Zeolite_Ag, PSy_Zeolite_Ag] = mapminmax(Y_Zeolite_Ag', 0, 1);

rng(42); 
cv = cvpartition(length(Y_Zeolite), 'HoldOut', 0.3);
train_idx = cv.training; test_idx = cv.test;
cv_ag = cvpartition(length(Y_Zeolite_Ag), 'HoldOut', 0.3);
train_idx_ag = cv_ag.training; test_idx_ag = cv_ag.test;

%% 4. COMPARATIVA DE 8 ALGORITMOS (Cálculo de todas las métricas)

algorithms = {'Fine Tree', 'Linear SVM', 'Narrow Neural Network', 'Medium Neural Network', ...
              'Wide Neural Network', 'Bilayered Neural Network', 'Trilayered Neural Network', 'Optimizable Neural Network'};

% Inicializar matrices para ZEOLITE
MAE_val_zeo = zeros(8,1); MSE_val_zeo = zeros(8,1); RMSE_val_zeo = zeros(8,1); R2_val_zeo = zeros(8,1);
MAE_test_zeo = zeros(8,1); MSE_test_zeo = zeros(8,1); RMSE_test_zeo = zeros(8,1); R2_test_zeo = zeros(8,1);

% Inicializar matrices para ZEOLITE-AG
MAE_val_ag = zeros(8,1); MSE_val_ag = zeros(8,1); RMSE_val_ag = zeros(8,1); R2_val_ag = zeros(8,1);
MAE_test_ag = zeros(8,1); MSE_test_ag = zeros(8,1); RMSE_test_ag = zeros(8,1); R2_test_ag = zeros(8,1);

% Entrenamiento: Zeolite
for i = 1:8
    switch i
        case 1, mdl = fitrtree(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'MinLeafSize', 4);
        case 2, mdl = fitrsvm(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'KernelFunction', 'linear', 'Standardize', true);
        case 3, mdl = fitrnet(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'LayerSizes', 10, 'Activations', 'relu', 'Standardize', true);
        case 4, mdl = fitrnet(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'LayerSizes', 25, 'Activations', 'relu', 'Standardize', true);
        case 5, mdl = fitrnet(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'LayerSizes', 100, 'Activations', 'relu', 'Standardize', true);
        case 6, mdl = fitrnet(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'LayerSizes', [10, 10], 'Activations', 'relu', 'Standardize', true);
        case 7, mdl = fitrnet(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'LayerSizes', [10, 10, 10], 'Activations', 'relu', 'Standardize', true);
        case 8, mdl = fitrnet(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations', 30, 'ShowPlots', false));
    end
    % Validación
    pred_v = mapminmax('reverse', predict(mdl, X_norm_Zeolite(:, train_idx)')', PSy_Zeolite)';
    Y_v = Y_Zeolite(train_idx);
    RMSE_val_zeo(i) = sqrt(mean((Y_v - pred_v).^2)); MSE_val_zeo(i) = mean((Y_v - pred_v).^2);
    MAE_val_zeo(i) = mean(abs(Y_v - pred_v)); R2_val_zeo(i) = 1 - sum((Y_v - pred_v).^2)/sum((Y_v - mean(Y_v)).^2);
    % Test
    pred_t = mapminmax('reverse', predict(mdl, X_norm_Zeolite(:, test_idx)')', PSy_Zeolite)';
    Y_t = Y_Zeolite(test_idx);
    RMSE_test_zeo(i) = sqrt(mean((Y_t - pred_t).^2)); MSE_test_zeo(i) = mean((Y_t - pred_t).^2);
    MAE_test_zeo(i) = mean(abs(Y_t - pred_t)); R2_test_zeo(i) = 1 - sum((Y_t - pred_t).^2)/sum((Y_t - mean(Y_t)).^2);
end

% Entrenamiento: Zeolite-Ag
for i = 1:8
    switch i
        case 1, mdl = fitrtree(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'MinLeafSize', 4);
        case 2, mdl = fitrsvm(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'KernelFunction', 'linear', 'Standardize', true);
        case 3, mdl = fitrnet(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'LayerSizes', 10, 'Activations', 'relu', 'Standardize', true);
        case 4, mdl = fitrnet(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'LayerSizes', 25, 'Activations', 'relu', 'Standardize', true);
        case 5, mdl = fitrnet(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'LayerSizes', 100, 'Activations', 'relu', 'Standardize', true);
        case 6, mdl = fitrnet(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'LayerSizes', [10, 10], 'Activations', 'relu', 'Standardize', true);
        case 7, mdl = fitrnet(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'LayerSizes', [10, 10, 10], 'Activations', 'relu', 'Standardize', true);
        case 8, mdl = fitrnet(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations', 30, 'ShowPlots', false));
    end
    % Validación
    pred_v = mapminmax('reverse', predict(mdl, X_norm_Zeolite_Ag(:, train_idx_ag)')', PSy_Zeolite_Ag)';
    Y_v = Y_Zeolite_Ag(train_idx_ag);
    RMSE_val_ag(i) = sqrt(mean((Y_v - pred_v).^2)); MSE_val_ag(i) = mean((Y_v - pred_v).^2);
    MAE_val_ag(i) = mean(abs(Y_v - pred_v)); R2_val_ag(i) = 1 - sum((Y_v - pred_v).^2)/sum((Y_v - mean(Y_v)).^2);
    % Test
    pred_t = mapminmax('reverse', predict(mdl, X_norm_Zeolite_Ag(:, test_idx_ag)')', PSy_Zeolite_Ag)';
    Y_t = Y_Zeolite_Ag(test_idx_ag);
    RMSE_test_ag(i) = sqrt(mean((Y_t - pred_t).^2)); MSE_test_ag(i) = mean((Y_t - pred_t).^2);
    MAE_test_ag(i) = mean(abs(Y_t - pred_t)); R2_test_ag(i) = 1 - sum((Y_t - pred_t).^2)/sum((Y_t - mean(Y_t)).^2);
end

%% 5. ENTRENAMIENTO DE LOS MODELOS ÓPTIMOS (Para Figuras 4 y 5)
net_Zeolite = fitrnet(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'LayerSizes', 25, 'Activations', 'relu', 'Standardize', true);
net_Zeolite_Ag = fitrnet(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'LayerSizes', 25, 'Activations', 'relu', 'Standardize', true);
tree_Zeolite = fitrtree(X_norm_Zeolite(:, train_idx)', Y_norm_Zeolite(train_idx)', 'MinLeafSize', 4);
tree_Zeolite_Ag = fitrtree(X_norm_Zeolite_Ag(:, train_idx_ag)', Y_norm_Zeolite_Ag(train_idx_ag)', 'MinLeafSize', 4);

%% 6. GENERACIÓN DE FIGURAS

% --- FIGURA 3: Superficies 3D de Rendimiento (Exacta al Preprint) ---
fig3 = figure('Position', [100 100 1200 1000], 'Color', 'white', 'Name', 'Figure 3: Surface Graphs of Model Performance');

% Crear la matriz Z (8 modelos x 8 métricas)
% Eje Y (Métricas de frente hacia atrás): MAE(Val), MSE(Val), RMSE(Val), R2(Val), MAE(Test), MSE(Test), RMSE(Test), R2(Test)
Z_zeo = [MAE_val_zeo, MSE_val_zeo, RMSE_val_zeo, R2_val_zeo, MAE_test_zeo, MSE_test_zeo, RMSE_test_zeo, R2_test_zeo]';
Z_ag = [MAE_val_ag, MSE_val_ag, RMSE_val_ag, R2_val_ag, MAE_test_ag, MSE_test_ag, RMSE_test_ag, R2_test_ag]';

[X, Y] = meshgrid(1:8, 1:8);

% Subplot Zeolite
subplot(2,1,1);
surf(X, Y, Z_zeo, 'EdgeColor', 'k', 'FaceAlpha', 0.9);
colormap(gca, turbo); shading interp;
xlabel('Models', 'FontWeight', 'bold'); ylabel('Metrics', 'FontWeight', 'bold'); zlabel('Values', 'FontWeight', 'bold');
title('Zeolite Performance Metrics', 'FontWeight', 'bold');
set(gca, 'XTick', 1:8, 'XTickLabel', {'Optimizable NN', 'Trilayered NN', 'Bilayered NN', 'Wide NN', 'Medium NN', 'Narrow NN', 'Linear SVM', 'Fine Tree'}, 'XTickLabelRotation', 30);
set(gca, 'YTick', 1:8, 'YTickLabel', {'MAE (Validation)', 'MSE (Validation)', 'RMSE (Validation)', 'R-Squared (Validation)', 'MAE (Test)', 'MSE (Test)', 'RMSE (Test)', 'R-Squared (Test)'});
view(45, 30); grid on; box on; colorbar;

% Subplot Zeolite-Ag
subplot(2,1,2);
surf(X, Y, Z_ag, 'EdgeColor', 'k', 'FaceAlpha', 0.9);
colormap(gca, turbo); shading interp;
xlabel('Models', 'FontWeight', 'bold'); ylabel('Metrics', 'FontWeight', 'bold'); zlabel('Values', 'FontWeight', 'bold');
title('Zeolite-Ag Performance Metrics', 'FontWeight', 'bold');
set(gca, 'XTick', 1:8, 'XTickLabel', {'Optimizable NN', 'Trilayered NN', 'Bilayered NN', 'Wide NN', 'Medium NN', 'Narrow NN', 'Linear SVM', 'Fine Tree'}, 'XTickLabelRotation', 30);
set(gca, 'YTick', 1:8, 'YTickLabel', {'MAE (Validation)', 'MSE (Validation)', 'RMSE (Validation)', 'R-Squared (Validation)', 'MAE (Test)', 'MSE (Test)', 'RMSE (Test)', 'R-Squared (Test)'});
view(45, 30); grid on; box on; colorbar;

savefig(fig3, 'Figure3_Surface_Graphs.fig'); exportgraphics(fig3, 'Figure3_Surface_Graphs.png', 'Resolution', 600);

% --- FIGURA 4: Predicciones Zeolite ---
fig4 = figure('Position', [100 100 1400 900], 'Color', 'white', 'Name', 'Figure 4: Zeolite Predictions');
pred_tree = mapminmax('reverse', predict(tree_Zeolite, X_norm_Zeolite')', PSy_Zeolite)';
subplot(2,2,1); plot(t_dense_Zeolite, pred_tree, 'go', 'MarkerSize', 6); hold on; plot(t_Zeolite, perc_Zeolite, 'bd', 'MarkerFaceColor', 'b'); 
xlabel('Time (min)'); ylabel('Cu(II) Adsorption (%)'); title('(a) Fine Tree Prediction Zeolite'); legend('Predicted Fine Tree', 'Experimental Data'); grid on;
subplot(2,2,2); scatter(Y_Zeolite, pred_tree, 50, 'g', 'filled'); hold on; plot([0 100], [0 100], 'k--'); xlabel('True Response'); ylabel('Predicted Response'); title('(b) Fine Tree: True vs Predicted'); grid on;
pred_nn = mapminmax('reverse', predict(net_Zeolite, X_norm_Zeolite')', PSy_Zeolite)';
subplot(2,2,3); plot(t_dense_Zeolite, pred_nn, 'go', 'MarkerSize', 6); hold on; plot(t_Zeolite, perc_Zeolite, 'bd', 'MarkerFaceColor', 'b'); 
xlabel('Time (min)'); ylabel('Cu(II) Adsorption (%)'); title('(c) Medium NN Prediction'); legend('Predicted Medium NN', 'Experimental Data'); grid on;
subplot(2,2,4); scatter(Y_Zeolite, pred_nn, 50, 'b', 'filled'); hold on; plot([0 100], [0 100], 'k--'); xlabel('True Response'); ylabel('Predicted Response'); title('(d) Medium NN: True vs Predicted'); grid on;
sgtitle('Figure 4: Comparison graphs of prediction models for Zeolite', 'FontSize', 16, 'FontWeight', 'bold');
savefig(fig4, 'Figure4_Zeolite_Predictions.fig'); exportgraphics(fig4, 'Figure4_Zeolite_Predictions.png', 'Resolution', 600);

% --- FIGURA 5: Predicciones Zeolite-Ag ---
fig5 = figure('Position', [100 100 1400 900], 'Color', 'white', 'Name', 'Figure 5: Zeolite-Ag Predictions');
pred_tree_ag = mapminmax('reverse', predict(tree_Zeolite_Ag, X_norm_Zeolite_Ag')', PSy_Zeolite_Ag)';
subplot(2,2,1); plot(t_dense_Zeolite_Ag, pred_tree_ag, 'go', 'MarkerSize', 6); hold on; plot(t_Zeolite_Ag, perc_Zeolite_Ag, 'bd', 'MarkerFaceColor', 'b'); 
xlabel('Time (min)'); ylabel('Cu(II) Adsorption (%)'); title('(a) Fine Tree Prediction Zeolite-Ag'); legend('Predicted Fine Tree', 'Experimental Data'); grid on;
subplot(2,2,2); scatter(Y_Zeolite_Ag, pred_tree_ag, 50, 'g', 'filled'); hold on; plot([0 100], [0 100], 'k--'); xlabel('True Response'); ylabel('Predicted Response'); title('(b) Fine Tree: True vs Predicted'); grid on;
pred_nn_ag = mapminmax('reverse', predict(net_Zeolite_Ag, X_norm_Zeolite_Ag')', PSy_Zeolite_Ag)';
subplot(2,2,3); plot(t_dense_Zeolite_Ag, pred_nn_ag, 'go', 'MarkerSize', 6); hold on; plot(t_Zeolite_Ag, perc_Zeolite_Ag, 'bd', 'MarkerFaceColor', 'b'); 
xlabel('Time (min)'); ylabel('Cu(II) Adsorption (%)'); title('(c) Medium NN Prediction'); legend('Predicted Medium NN', 'Experimental Data'); grid on;
subplot(2,2,4); scatter(Y_Zeolite_Ag, pred_nn_ag, 50, 'b', 'filled'); hold on; plot([0 100], [0 100], 'k--'); xlabel('True Response'); ylabel('Predicted Response'); title('(d) Medium NN: True vs Predicted'); grid on;
sgtitle('Figure 5: Comparison graphs of prediction models for Zeolite-Ag', 'FontSize', 16, 'FontWeight', 'bold');
savefig(fig5, 'Figure5_ZeoliteAg_Predictions.fig'); exportgraphics(fig5, 'Figure5_ZeoliteAg_Predictions.png', 'Resolution', 600);

%% 7. GUARDAR RESULTADOS
Algorithm = algorithms'; 
T1 = table(Algorithm, RMSE_test_zeo, MAE_test_zeo, R2_test_zeo, RMSE_test_ag, MAE_test_ag, R2_test_ag, ...
    'VariableNames', {'Algorithm', 'RMSE_Zeolite', 'MAE_Zeolite', 'R2_Zeolite', 'RMSE_Zeolite_Ag', 'MAE_Zeolite_Ag', 'R2_Zeolite_Ag'});
writetable(T1, 'Table1_Algorithm_Comparison.csv');

fprintf('\n✅ Ejecución completada. Figuras y Tabla generadas correctamente.\n');