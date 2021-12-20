% Machine Learning Coursework 2020
% Classification Models To Classify White Wine Quality
% By: Jiseong Park
% Model: Random Forests

clear all;
rng('default');

wine_table = readtable('WhiteWineQuality.csv');
wine = wine_table{:, :}; % Converting from table to matrix.

% ----------Data Preprocessing----------
% ----------Checking For Missing Data----------

missing = sum(ismissing(wine)); % No missing values found.

% ----------Dealing with Outliers----------

[data, TF] = rmoutliers(wine, 'median'); % Removing values more than three scaled MAD away from the median.

% ----------Feature Scaling----------

feat = data(:, 1:11); % Extracting features from the wine matrix.

% Normalising features to between 0 and 1.
for i=1:11
    feat(:, i) = (feat(:, i) - min(feat(:, i))) / (max(feat(:, i)) - min(feat(:, i)));
end

poorIdx = data(:,12) < 6;
data(poorIdx, 12) = 0;
goodIdx = data(:, 12) >= 6;
data(goodIdx, 12) = 1;

% ----------Initial Decision Tree Test---------- 

target = data(:, 12);

% Training and test percentage split
p = 0.80;

[m, n] = size(feat);
[r, c] = size(target);

shuffled_idx = randperm(m);

train_x = feat(shuffled_idx(1:round(p * m)), :); 
train_y = target(shuffled_idx(1:round(p * r)), :);

test_x = feat(shuffled_idx(round(p * m) + 1:end), :);
test_y = target(shuffled_idx(round(p * r) + 1:end), :);

% First we build a simple decision tree and do a quick evaluation with test data.
dTree = fitctree(train_x, train_y);

% ----------Random Forest---------- 

NumTrees = 64;

rForest = TreeBagger(NumTrees, train_x, train_y, 'OOBPrediction', 'on',...
               'Method', 'classification');

% Hyperparameter Tuned

maxMinLS = 20; % Minimum number of trees per leaf.
minLS = optimizableVariable('minLS', [1, maxMinLS], 'Type', 'integer');
numPTS = optimizableVariable('numPTS', [1, size(train_x, 2)-1], 'Type', 'integer');
hyperparametersRF = [minLS; numPTS];

optimised = bayesopt(@(params)oobMCR_RF(params, train_x, train_y, NumTrees), hyperparametersRF,...
    'AcquisitionFunctionName', 'expected-improvement', 'Verbose', 0);

bestOOBMCR = optimised.MinObjective;
bestHyperparameters = optimised.XAtMinObjective;

rForestHyp = TreeBagger(NumTrees, train_x, train_y, 'OOBPrediction', 'on',...
                  'Method', 'classification',...
                  'MinLeafSize', bestHyperparameters.minLS,...
                  'NumPredictorstoSample', bestHyperparameters.numPTS);

% ----------Validation----------

err = crossval('mcr', train_x, train_y, 'Predfun', @classf, 'Stratify', train_y);

figure;
oobErrorBaggedEnsembleRF = oobError(rForest);
plot(oobErrorBaggedEnsembleRF);
xlabel 'Number of grown trees for Random Forest';
ylabel 'Out-of-bag classification error for Random Forest';

% Hyperparameter tuned

errHyp = crossval('mcr', train_x, train_y, 'Predfun', @classfhyp, 'Stratify', train_y);

figure;
oobErrorBaggedEnsembleRFHyp = oobError(rForestHyp);
plot(oobErrorBaggedEnsembleRFHyp);
xlabel 'Number of grown trees for Tuned Random Forest';
ylabel 'Out-of-bag classification error for Tuned Random Forest';

% ----------Analysing Training Predictions----------

% Original

[rf_pred_trained, score_rf_trained] = predict(rForest, train_x);
rf_pred_trained = str2double(rf_pred_trained); 
rf_results_trained = confusionmat(train_y, rf_pred_trained);
rf_sumResults_trained = sum(sum(rf_results_trained));
rf_acc_trained = (rf_results_trained(1, 1) + rf_results_trained(2, 2))/rf_sumResults_trained;

% Hyperparameter Tuned

[rfHyp_pred_trained, score_rfHyp_trained] = predict(rForestHyp, train_x);
rfHyp_pred_trained = str2double(rfHyp_pred_trained); 
rfHyp_results_trained = confusionmat(train_y, rfHyp_pred_trained);
rfHyp_acc_trained = (rfHyp_results_trained(1, 1) + rfHyp_results_trained(2, 2))/rf_sumResults_trained;

% ----------Cross Validation Function----------

% Original

function pred = classf(train_x, train_y, test_x)

trees = 64;

mdl = TreeBagger(trees, train_x, train_y, 'OOBPrediction', 'on',...
                'Method', 'classification');
            
pred = predict(mdl, test_x);
end

% Hyperparameter Tuned

function pred = classfhyp(train_x, train_y, test_x)

trees = 64;

maxMinLS = 20;
minLS = optimizableVariable('minLS', [1, maxMinLS], 'Type', 'integer');
numPTS = optimizableVariable('numPTS', [1, size(train_x, 2)-1], 'Type', 'integer');
hyperparametersRF = [minLS; numPTS];

opt = bayesopt(@(params)oobMCR_RF(params, train_x, train_y, trees), hyperparametersRF,...
    'AcquisitionFunctionName', 'expected-improvement', 'Verbose', 0);

bestHyperparameters = opt.XAtMinObjective;

mdl = TreeBagger(trees, train_x, train_y, 'OOBPrediction', 'on',...
                'Method', 'classification',...
                'MinLeafSize', bestHyperparameters.minLS,...
                'NumPredictorstoSample', bestHyperparameters.numPTS);
pred = predict(mdl, test_x);
end

% ----------Hyperparameter Tuning Function----------

% The objective function below returns the weighted misclassification rate
% of a random forest model that has been trained using the training data
% and the optimised parameters...
function oobMCR = oobMCR_RF(params, X, Y, trees)
    % oobMCR_RF trains the random forest and estimates out-of-bag error
    % (oobError). 
    % oobMCR trains a random forest using the training features data, X, 
    % and the parameter specifications in params, and then returns the 
    % oobError, which computes the misclassification probability. 
    % params is an array of optimizableVariable objects corresponding to
    % the minimum leaf size and number of predictors to sample at each node.
    ranForest = TreeBagger(trees, X, Y, 'OOBPrediction', 'on',...
                      'Method', 'classification',...
                      'MinLeafSize', params.minLS,...
                      'NumPredictorstoSample', params.numPTS);
    oobMCR = oobError(ranForest, 'Mode','ensemble');
end
