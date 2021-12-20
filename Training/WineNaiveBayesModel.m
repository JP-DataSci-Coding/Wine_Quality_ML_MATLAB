% Machine Learning Coursework 2020
% Classification Models To Classify White Wine Quality
% By: Jiseong Park
% Model: Naive Bayes

clear all;
rng('default'); % For reproducibility.

wine_table = readtable('WhiteWineQuality.csv');
wine = wine_table{:, :}; % Converting from table to matrix.

% ----------Data Preprocessing----------
% ----------Checking For Missing Data----------

missing = sum(ismissing(wine)); % No missing values found.

% ----------Dealing with Outliers----------

[data, TF] = rmoutliers(wine, 'median'); % Removing values more than three scaled MAD away from the median.

descriptive_stats = table();
descriptive_stats.Min = min(data(:, 1:11))';
descriptive_stats.Max = max(data(:, 1:11))';
descriptive_stats.Mean = mean(data(:, 1:11))';
descriptive_stats.Std_dev = std(data(:, 1:11))';

% ----------Feature Scaling----------

feat = data(:, 1:11); % Extracting features from the wine matrix.

% Normalising features to between 0 and 1.
for i=1:11
    feat(:, i) = (feat(:, i) - min(feat(:, i))) / (max(feat(:, i)) - min(feat(:, i)));
end

% ----------Dealing with Categorical Target----------

% The 'quality' column is the target of this dataset. 
% It takes 9 values, 1 to 10.
quality = data(:, 12);

figure;
[QualityGroupCounts, QualityGroup] = groupcounts(quality); 
bar(QualityGroup, QualityGroupCounts);
title('Wine Quality Pre-Grouping', 'fontweight', 'bold', 'fontsize', 16);

% The majority of quality values lie between 5 and 7, with very low counts
% for 3, 4, 8 and 9. To make sure that our classifier algorithms are working  
% with robust labels, they will be grouped as such:
poorIdx = data(:, 12) < 6;
data(poorIdx, 12) = 0;
goodIdx = data(:, 12) >= 6;
data(goodIdx, 12) = 1;

figure;
[NewQualityGroupCounts, NewQualityGroup] = groupcounts(data(:, 12)); 
bar(NewQualityGroup, NewQualityGroupCounts,  'r');
title('Wine Quality Post-Grouping', 'fontweight', 'bold', 'fontsize', 16);

% ----------Feature Distribution Checks----------

target = data(:, 12);

featureNames = wine_table.Properties.VariableNames;

for i = 1:11
    figure;
    histogram(feat(:, i));
    title(featureNames(:, i), 'fontweight', 'bold', 'fontsize', 16);
end

% With the exception of residual sugar, the other features at the very
% least, have a rough Normal distribution.

% Log transformation check of residual sugar.
figure;
histogram(log10(feat(:, 4)));
title(featureNames(:, 4), 'fontweight', 'bold', 'fontsize', 16);

% Residual sugar has a distribution that is far closer to Normal than
% before:
feat(:, 4) = log10(feat(:, 4));

% ----------Naive Bayes----------

% Test percentage split:
p = 0.8;

[m, n] = size(feat);
[r, c] = size(target);

shuffled_idx = randperm(m);

train_x = feat(shuffled_idx(1:round(p * m)), :); 
train_y = target(shuffled_idx(1:round(p * r)), :);

test_x = feat(shuffled_idx(round(p * m) + 1:end), :);
test_y = target(shuffled_idx(round(p * r) + 1:end), :);

% Multinomial distribution has been ignored as the features have a continuous
% nature and because the data has negative values due to the log transformation
% of the residual sugar variable.

% Gaussian

naiveBayesModel = fitcnb(train_x, train_y, 'Distribution', 'normal');

% Kernel

naiveBayesModelKernel = fitcnb(train_x, train_y, 'Distribution', 'kernel');

% Hyperparameter Tuned - after basic model runs (see below).

optWidth = optimizableVariable('optWidth', [0, 1], 'Type', 'real');
hyperparametersNB = optWidth;

optimised = bayesopt(@(params)cvLoss_NB(params, train_x, train_y), hyperparametersNB,...
    'AcquisitionFunctionName', 'expected-improvement', 'Verbose', 0);

bestCVLoss = optimised.MinObjective;
bestHyperparameters = optimised.XAtMinObjective;

naiveBayesModelHyp = fitcnb(train_x, train_y, 'Distribution', 'kernel',...
                            'Width', table2array(bestHyperparameters));

% ----------Validation----------

% Initial Model Assessments Begin - The hypothesis is that the Gaussian version
% will perform best.

cvNaiveBayesModel = crossval(naiveBayesModel);
cvNaiveBayesModelLoss = kfoldLoss(cvNaiveBayesModel);

cvNaiveBayesModelKernel = crossval(naiveBayesModelKernel);
cvNaiveBayesModelKernelLoss = kfoldLoss(cvNaiveBayesModelKernel);

cvNaiveBayesModelHyp = crossval(naiveBayesModelHyp);
cvNaiveBayesModelHypLoss = kfoldLoss(cvNaiveBayesModelHyp);

% ----------Analysing Training Accuracy----------

% Gaussian

[predictions_trained, score_nb_trained] = predict(naiveBayesModel, train_x);
results_trained = confusionmat(train_y, predictions_trained);
sumResults_trained = sum(sum(results_trained));
accuracy_trained = (results_trained(1, 1) + results_trained(2, 2))/sumResults_trained;

% Kernel 

[predKernel_trained, score_nbKernel_trained] = predict(naiveBayesModelKernel, train_x);
resultsKernel_trained = confusionmat(train_y, predKernel_trained);
accuracyKernel_trained = (resultsKernel_trained(1, 1) + resultsKernel_trained(2, 2))/sumResults_trained; 

% Suprisingly the kernel version outperformed the Gaussian.
                         
% Hyperparameter Tuned - after basic model runs.

[hyp_predictions_trained, score_hypNb_trained] = predict(naiveBayesModelHyp, train_x);
hyp_results_trained = confusionmat(train_y, hyp_predictions_trained);
hyp_accuracy_trained = (hyp_results_trained(1, 1) + hyp_results_trained(2, 2))/sumResults_trained;

% ----------Hyperparameter Tuning Function----------

function cvLoss = cvLoss_NB(params, X, Y)
    optWidth = table2array(params);
    nb = fitcnb(X, Y, 'Distribution', 'kernel',...
                'Width', optWidth);
                     
    cvNb = crossval(nb);
    cvLoss = kfoldLoss(cvNb);
end