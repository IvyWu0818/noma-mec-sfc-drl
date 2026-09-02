% plot_sac_training_v20.m
% MATLAB equivalent of experiments/plot_sac_training_v20.py
% Reads results/sac_v20_training_metrics.json and plots SAC V20 training curves.
%
% Run from repo root:  matlab -batch "run('matlab/plot_sac_training_v20.m')"
% or open in MATLAB and press Run.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);   % assumes matlab/ is one level under repo root
cd(repoRoot);

inputFile   = fullfile('results', 'sac_v20_training_metrics.json');
outputDir   = fullfile('results', 'figures_v20_matlab');
td3v17File  = fullfile('results', 'td3_v17_training_metrics.json');
sacv18File  = fullfile('results', 'sac_v18_training_metrics.json');
smoothWin   = 15;

if ~isfile(inputFile)
    error('Cannot find %s -- run train_sac_v20.py first', inputFile);
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

data = jsondecode(fileread(inputFile));
nEp  = numel(data.episode_rewards);
eps  = (0:nEp-1)';

%% ---- 1. Individual metric plots -----------------------------------
singleCfg = {
    'episode_rewards',                'Cumulative Reward',          'v20_reward.png',            [0.27 0.51 0.71];
    'episode_avg_delay',              'Avg Delay (ms)',              'v20_delay.png',             [1.00 0.27 0.00];
    'episode_avg_slack',              'Avg Slack (ms)',              'v20_slack.png',             [0.17 0.63 0.17];
    'episode_timeout_ratio',          'Timeout Ratio',               'v20_timeout.png',           [0.00 0.00 0.00];
    'episode_cpu_viol_rate',          'CPU Violation Rate',          'v20_cpu_viol_rate.png',     [0.50 0.00 0.50];
    'episode_avg_deadline_pressure',  'Deadline Pressure',           'v20_deadline_pressure.png', [0.00 0.39 0.00];
    'episode_task_mix_urgent_ratio',  'Urgent Task Ratio',           'v20_urgent_ratio.png',      [0.84 0.15 0.16];
    'episode_avg_sinr_db',            'Avg SINR (dB)',               'v20_sinr.png',              [0.10 0.44 0.66];
    'episode_avg_channel_rate',       'Avg Channel Rate R (Mbps)',   'v20_channel_rate.png',      [0.06 0.43 0.34];
    'episode_channel_overflow_ratio', 'Channel Overflow Ratio',      'v20_ch_overflow.png',       [0.85 0.35 0.19];
    'episode_avg_channel_entropy',    'Channel Assignment Entropy',  'v20_ch_entropy.png',        [0.50 0.47 0.87];
    'episode_avg_rho',                'Avg Offload Ratio \rho',      'v20_rho.png',                [0.72 0.53 0.04];
    'episode_avg_queue_delta',        'Avg Queue Delta |dq|',        'v20_queue_delta.png',       [0.55 0.27 0.07];
};

for i = 1:size(singleCfg, 1)
    key   = singleCfg{i, 1};
    ylab  = singleCfg{i, 2};
    fname = singleCfg{i, 3};
    col   = singleCfg{i, 4};

    if ~isfield(data, key)
        continue;
    end
    y = data.(key);
    ySmooth = movmean(y, smoothWin, 'Endpoints', 'shrink');

    fig = figure('Visible', 'off', 'Position', [100 100 900 450]);
    plot(eps, y, '-', 'Color', [0.83 0.83 0.83 0.35], 'LineWidth', 0.8); hold on;
    plot(eps, ySmooth, '-', 'Color', col, 'LineWidth', 2.0);
    title(sprintf('SAC V20 Training -- %s', ylab), 'FontSize', 13);
    xlabel('Episode'); ylabel(ylab);
    legend({'raw', sprintf('smoothed (w=%d)', smoothWin)}, 'FontSize', 10, 'Location', 'best');
    grid on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);

    saveas(fig, fullfile(outputDir, fname));
    saveas(fig, fullfile(outputDir, strrep(fname, '.png', '.svg')));
    close(fig);
    fprintf('saved %s\n', fname);
end

%% ---- 2. Delay decomposition stacked area ---------------------------
decompKeys   = {'episode_avg_t_ul', 'episode_avg_t_comp', 'episode_avg_t_link'};
decompLabels = {'Upload t_{ul} (NOMA x rho)', 'Compute t_{comp}', 'Link t_{link}'};
decompColors = [0.30 0.61 0.91; 0.91 0.48 0.30; 0.43 0.75 0.40];

if all(isfield(data, decompKeys))
    stackVals = zeros(nEp, numel(decompKeys));
    for k = 1:numel(decompKeys)
        stackVals(:, k) = movmean(data.(decompKeys{k}), smoothWin, 'Endpoints', 'shrink');
    end

    fig = figure('Visible', 'off', 'Position', [100 100 1100 450]);
    h = area(eps, stackVals);
    for k = 1:numel(decompKeys)
        h(k).FaceColor = decompColors(k, :);
        h(k).FaceAlpha = 0.75;
    end
    title('SAC V20 Training -- Delay Decomposition (smoothed)', 'FontSize', 13);
    xlabel('Episode'); ylabel('Avg Delay (ms)');
    legend(decompLabels, 'Location', 'northeast', 'FontSize', 10);
    grid on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.4);

    saveas(fig, fullfile(outputDir, 'v20_delay_decomp.png'));
    saveas(fig, fullfile(outputDir, 'v20_delay_decomp.svg'));
    close(fig);
    fprintf('saved v20_delay_decomp.png\n');
end

%% ---- 3. NOMA 4-panel ------------------------------------------------
nomaCfg = {
    'episode_avg_sinr_db',            'Avg SINR (dB)',           [0.10 0.44 0.66];
    'episode_avg_channel_rate',       'Avg Channel Rate (Mbps)', [0.06 0.43 0.34];
    'episode_channel_overflow_ratio', 'Channel Overflow Ratio',  [0.85 0.35 0.19];
    'episode_avg_channel_entropy',    'Channel Entropy',         [0.50 0.47 0.87];
};

fig = figure('Visible', 'off', 'Position', [100 100 1300 550]);
tl = tiledlayout(2, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
for idx = 1:size(nomaCfg, 1)
    key = nomaCfg{idx, 1};
    if ~isfield(data, key)
        continue;
    end
    ylab = nomaCfg{idx, 2};
    col  = nomaCfg{idx, 3};
    y = data.(key);
    ySmooth = movmean(y, smoothWin, 'Endpoints', 'shrink');

    nexttile;
    plot(eps, y, '-', 'Color', [0.83 0.83 0.83 0.3], 'LineWidth', 0.7); hold on;
    plot(eps, ySmooth, '-', 'Color', col, 'LineWidth', 2.0);
    title(ylab, 'FontSize', 11);
    xlabel('Episode', 'FontSize', 9); ylabel(ylab, 'FontSize', 9);
    grid on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.4);
end
title(tl, 'SAC V20 -- NOMA Channel Metrics (overflow penalty=20)', ...
    'FontSize', 13, 'FontWeight', 'bold');

saveas(fig, fullfile(outputDir, 'v20_noma_panel.png'));
saveas(fig, fullfile(outputDir, 'v20_noma_panel.svg'));
close(fig);
fprintf('saved v20_noma_panel.png\n');

%% ---- 4. Main 6-panel overview ---------------------------------------
panelCfg = {
    'episode_rewards',                'Cumulative Reward',      [0.27 0.51 0.71];
    'episode_avg_delay',              'Avg Delay (ms)',         [1.00 0.27 0.00];
    'episode_timeout_ratio',          'Timeout Ratio',          [0.41 0.41 0.41];
    'episode_cpu_viol_rate',          'CPU Violation Rate',     [0.50 0.00 0.50];
    'episode_avg_rho',                'Avg Offload Ratio \rho',  [0.72 0.53 0.04];
    'episode_channel_overflow_ratio', 'Channel Overflow Ratio', [0.85 0.35 0.19];
};

fig = figure('Visible', 'off', 'Position', [100 100 1500 750]);
tl = tiledlayout(2, 3, 'TileSpacing', 'loose', 'Padding', 'compact');
for idx = 1:size(panelCfg, 1)
    key = panelCfg{idx, 1};
    if ~isfield(data, key)
        continue;
    end
    ylab = panelCfg{idx, 2};
    col  = panelCfg{idx, 3};
    y = data.(key);
    ySmooth = movmean(y, smoothWin, 'Endpoints', 'shrink');

    nexttile;
    plot(eps, y, '-', 'Color', [0.83 0.83 0.83 0.3], 'LineWidth', 0.7); hold on;
    plot(eps, ySmooth, '-', 'Color', col, 'LineWidth', 2.0);
    title(ylab, 'FontSize', 10);
    xlabel('Episode', 'FontSize', 8); ylabel(ylab, 'FontSize', 8);
    grid on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.4);
end
title(tl, 'SAC V20 Training Metrics', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(outputDir, 'v20_panel.png'));
saveas(fig, fullfile(outputDir, 'v20_panel.svg'));
close(fig);
fprintf('saved v20_panel.png\n');

%% ---- 5. TD3 V17 vs SAC V18 vs SAC V20 comparison ---------------------
compareCfg = {
    'episode_rewards',                'Cumulative Reward';
    'episode_avg_delay',              'Avg Delay (ms)';
    'episode_avg_t_link',             'Avg t_{link} (ms)';
    'episode_channel_overflow_ratio', 'Channel Overflow Ratio';
    'episode_cpu_viol_rate',          'CPU Violation Rate';
    'episode_avg_rho',                'Avg Offload Ratio \rho';
};

loaded = struct();
if isfile(td3v17File)
    loaded.TD3_V17 = jsondecode(fileread(td3v17File));
end
if isfile(sacv18File)
    loaded.SAC_V18 = jsondecode(fileread(sacv18File));
end

verNames  = {'TD3_V17', 'SAC_V18', 'SAC_V20'};
verLabels = {'TD3 V17', 'SAC V18', 'SAC V20'};
verColors = [0.94 0.50 0.50; 0.27 0.51 0.71; 0.18 0.55 0.34];
verStyles = {'--', '-.', '-'};

fig = figure('Visible', 'off', 'Position', [100 100 1700 950]);
tl = tiledlayout(2, 3, 'TileSpacing', 'loose', 'Padding', 'compact');
for idx = 1:size(compareCfg, 1)
    key  = compareCfg{idx, 1};
    ylab = compareCfg{idx, 2};

    nexttile; hold on;
    legendEntries = {};
    for v = 1:2  % TD3_V17, SAC_V18 (if present)
        vname = verNames{v};
        if isfield(loaded, vname) && isfield(loaded.(vname), key)
            vy = loaded.(vname).(key);
            ve = (0:numel(vy)-1)';
            vySmooth = movmean(vy, smoothWin, 'Endpoints', 'shrink');
            plot(ve, vySmooth, verStyles{v}, 'Color', verColors(v, :), ...
                'LineWidth', 1.5);
            legendEntries{end+1} = verLabels{v}; %#ok<SAGROW>
        end
    end
    if isfield(data, key)
        ySmooth = movmean(data.(key), smoothWin, 'Endpoints', 'shrink');
        plot(eps, ySmooth, verStyles{3}, 'Color', verColors(3, :), 'LineWidth', 2.2);
        legendEntries{end+1} = 'SAC V20'; %#ok<SAGROW>
    end
    title(ylab, 'FontSize', 10);
    xlabel('Episode', 'FontSize', 8); ylabel(ylab, 'FontSize', 8);
    legend(legendEntries, 'FontSize', 9, 'Location', 'best');
    grid on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.4);
end
title(tl, {'TD3 V17 vs SAC V18 vs SAC V20 -- Training Comparison', ...
    '(reward\_scale identical across all three -- same env, same scale)'}, ...
    'FontSize', 13, 'FontWeight', 'bold');

saveas(fig, fullfile(outputDir, 'td3_v17_sac_v18_v20_compare.png'));
saveas(fig, fullfile(outputDir, 'td3_v17_sac_v18_v20_compare.svg'));
close(fig);
fprintf('saved td3_v17_sac_v18_v20_compare.png\n');

%% ---- 6. Numeric summary (last 20 episodes) ---------------------------
fprintf('\nSAC V20 Training Summary (last 20 episodes mean +/- std):\n');
reportCfg = {
    'episode_avg_delay',              'Avg Delay (ms)       ';
    'episode_avg_slack',              'Avg Slack (ms)       ';
    'episode_timeout_ratio',          'Timeout Ratio        ';
    'episode_cpu_viol_rate',          'CPU Violation Rate   ';
    'episode_avg_deadline_pressure',  'Deadline Pressure    ';
    'episode_task_mix_urgent_ratio',  'Urgent Task Ratio    ';
    'episode_avg_t_ul',               'Avg t_ul (ms)        ';
    'episode_avg_t_comp',             'Avg t_comp (ms)      ';
    'episode_avg_t_link',             'Avg t_link (ms)      ';
    'episode_avg_sinr_db',            'Avg SINR (dB)        ';
    'episode_avg_channel_rate',       'Avg Channel Rate     ';
    'episode_channel_overflow_ratio', 'Channel Overflow Ratio';
    'episode_avg_channel_entropy',    'Channel Entropy      ';
    'episode_avg_rho',                'Avg Rho              ';
    'episode_avg_queue_delta',        'Avg Queue Delta      ';
    'episode_rewards',                'Avg Reward           ';
};
for i = 1:size(reportCfg, 1)
    key = reportCfg{i, 1};
    lab = reportCfg{i, 2};
    if isfield(data, key) && numel(data.(key)) >= 20
        tail = data.(key)(end-19:end);
        fprintf('  %s: %.4f  +/- %.4f\n', lab, mean(tail), std(tail));
    end
end

fprintf('\nAll figures saved to %s\n', outputDir);