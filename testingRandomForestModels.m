clear all;
rng('default');

rfModels = load('trainedRandomForestModels.mat');

dTree = rfModels.dTree;
rF_Original = rfModels.rForest;
rF_Tuned = rfModels.rForestHyp;

% ----------Load Test Data----------

test_x = readtable('test_x.csv');
test_x = test_x{:, :}; % Converting from table to matrix.

test_y = readtable('test_y.csv');
test_y = test_y{:, :}; % Converting from table to matrix.

% ----------Test Models----------

% First we evaluate a basic decision tree with test data.

[dt_pred, score_dt] = predict(dTree, test_x);
dt_results = confusionmat(test_y, dt_pred);
dt_sumResults = sum(sum(dt_results));
dt_acc = (dt_results(1, 1) + dt_results(2, 2))/dt_sumResults;

% Random Forest.

[pred, score] = predict(rF_Original, test_x);
pred = str2double(pred); 
res = confusionmat(test_y, pred);
sumResults = sum(sum(res));
accuracy = (res(1, 1) + res(2, 2))/sumResults;

[X_rf, Y_rf, T_rf, AUC_rf] = perfcurve(test_y, score(:, 2), 1); 

% Random Forest Tuned.

[predTuned, score_tuned] = predict(rF_Tuned, test_x);
predTuned = str2double(predTuned); 
resTuned = confusionmat(test_y, predTuned);
accuracyTuned = (resTuned(1, 1) + resTuned(2, 2))/sumResults;

[X_rfTuned, Y_rfTuned, T_rfTuned, AUC_rfTuned] = perfcurve(test_y, score_tuned(:, 2), 1); 

figure;
plot(X_rf,Y_rf);
hold on;
plot(X_rfTuned,Y_rfTuned);
legend('Random Forest', 'Random Forest Hyperparameter Tuned', 'Location', 'Best');
xlabel('False positive rate'); ylabel('True positive rate');
title("Random Forest ROC - Original AUC = " + num2str(AUC_rf) + ", Hyp AUC = " + num2str(AUC_rfTuned));
hold off;
