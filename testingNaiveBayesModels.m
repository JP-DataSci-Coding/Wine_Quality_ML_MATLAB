clear all;
rng('default');

nbModels = load('trainedNaiveBayesModels.mat');

nb_Gaussian = nbModels.naiveBayesModel;
nb_Original = nbModels.naiveBayesModelKernel;
nb_Tuned = nbModels.naiveBayesModelHyp;

% ----------Load Test Data----------

test_x = readtable('test_x.csv');
test_x = test_x{:, :}; % Converting from table to matrix.

test_y = readtable('test_y.csv');
test_y = test_y{:, :}; % Converting from table to matrix.

% ----------Test Models----------

% Gaussian. 

[pred_Gaussian, score_Gaussian] = predict(nb_Gaussian, test_x);
resGaussian = confusionmat(test_y, pred_Gaussian);
sumResults = sum(sum(resGaussian));
accuracyGaussian = (resGaussian(1, 1) + resGaussian(2, 2))/sumResults;  

% Original (Kernel).

[pred_Original, score] = predict(nb_Original, test_x);
resOriginal = confusionmat(test_y, pred_Original);
accuracy = (resOriginal(1, 1) + resOriginal(2, 2))/sumResults;  

[X_nb, Y_nb, T_nb, AUC_nb] = perfcurve(test_y,...
                             score(:, logical(nb_Original.ClassNames)),...
                             1); 

% Tuned.
                         
[predTuned, score_tuned] = predict(nb_Tuned, test_x);
resTuned = confusionmat(test_y, predTuned);
accuracyTuned = (resTuned(1, 1) + resTuned(2, 2))/sumResults;

[X_nbTuned, Y_nbTuned, T_nbTuned, AUC_nbTuned] = perfcurve(test_y,...
                                         score_tuned(:, logical(nb_Tuned.ClassNames)),...
                                         1);
                                     
figure;
plot(X_nb, Y_nb);
hold on;
plot(X_nbTuned, Y_nbTuned);
legend('Naive Bayes', 'Naive Bayes Hyperparameter Tuned', 'Location', 'Best');
xlabel('False positive rate'); ylabel('True positive rate');
title("Naive Bayes ROC - Original AUC = " + num2str(AUC_nb) + ", Hyp AUC = " + num2str(AUC_nbTuned));
hold off;