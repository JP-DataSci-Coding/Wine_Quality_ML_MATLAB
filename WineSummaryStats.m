% Machine Learning Coursework 2020
% Classification Models To Classify White Wine Quality
% By: Jiseong Park
% Summary Statistics

clear all;

% The white wine dataset has been chosen as it has more observations.
wine_table = readtable('WhiteWineQuality.csv');
wine = wine_table{:, :} % Converting from table to matrix.

feat = wine(:, 1:11);
target = wine(:, 12);

% ----------Descriptive Statistics----------

descriptive_stats = table();
descriptive_stats.Min = min(feat)';
descriptive_stats.Max = max(feat)';
descriptive_stats.Mean = mean(feat)';
descriptive_stats.Std_dev = std(feat)';

% ----------Feature Distribution Checks----------

featureNames = wine_table.Properties.VariableNames;

for i = 1:11
    figure;
    histogram(feat(:, i));
    title(featureNames(:, i), 'fontweight', 'bold', 'fontsize', 16);
end

% ----------Box Plot Checks----------

for i = 1:11
    figure;
    boxplot(feat(:, i));
    title(featureNames(:, i), 'fontweight', 'bold', 'fontsize', 16);
end

% ----------Feature Correlation Checks----------

corrFeat = corrcoef(feat);