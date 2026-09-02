% plot_ieee_iotj_figs.m
% Reproduces the 9 IEEE IoTJ paper figures (fig1_system_arch ... fig9_runtime)
% in MATLAB, from this repo's existing experiment outputs. Each figure is
% saved as both .png (300 dpi) and .svg (vector).
%
% Data provenance (matches experiments/plot_final_compare_v20.py and
% experiments/plot_drl_multiseed_v18_ppo.py, which produced the originals):
%   fig1_system_arch    -- hand-drawn schematic, no data source
%   fig2_reward          results/final_compare_v20.json  (SAC/TD3/PPO only)
%   fig3_seeds            results/{sac,td3,ppo}_v17_training_metrics.json
%                          results/{sac,td3,ppo}_v18_seed{1,2}_training_metrics.json
%   fig4_e2e_delay        results/final_compare_v20.json  (5-way)
%   fig5_delay_decomp     results/final_compare_v20.json  (5-way, stacked)
%   fig6_cpu_violation    results/final_compare_v20.json  (5-way)
%   fig7_channel_overflow results/final_compare_v20.json  (5-way)
%   fig8_timeout          results/final_compare_v20.json  (5-way)
%   fig9_runtime          results/final_compare_v20.json  (5-way, log y)
%
% Run from repo root:  matlab -batch "run('matlab/plot_ieee_iotj_figs.m')"
% or open in MATLAB and press Run. Requires R2016b+ (script local functions).

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);   % assumes matlab/ is one level under repo root
cd(repoRoot);

% Change this to write straight into the paper's figs/ folder if desired, e.g.:
%   outputDir = '/Users/ivywu/Documents/專題/IEEE IoTJ/figs';
outputDir = fullfile('results', 'figures_paper');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% ---- shared color palette (matches experiments/plot_final_compare_v20.py) ----
COL = struct( ...
    'SAC',    hex2rgb('#7E57C2'), ...
    'TD3',    hex2rgb('#2E86AB'), ...
    'PPO',    hex2rgb('#82C882'), ...
    'Greedy', hex2rgb('#E07B54'), ...
    'GA',     hex2rgb('#C2A83E'));

DECOMP_COL = [hex2rgb('#7DC3E8'); hex2rgb('#F0A070'); hex2rgb('#82C882')];

algos5 = {'SAC', 'TD3', 'PPO', 'Greedy', 'GA'};
algos3 = {'SAC', 'TD3', 'PPO'};

%% ---- Fig 1: system architecture (schematic, no data) --------------------
plot_fig1_system_arch(outputDir);

%% ---- Load final_compare_v20.json for fig2, fig4-fig9 --------------------
evalFile = fullfile('results', 'final_compare_v20.json');
if ~isfile(evalFile)
    error('Missing %s -- run "python -m experiments.eval_final_compare_v20" first', evalFile);
end
data = jsondecode(fileread(evalFile));

%% ---- Fig 2: Cumulative Reward (DRL only) ---------------------------------
plot_bar_metric(data, algos3, COL, 'episode_rewards', ...
    'Cumulative Reward (DRL only)', 'Reward', ...
    fullfile(outputDir, 'fig2_reward'));

%% ---- Fig 4: End-to-End Delay ---------------------------------------------
plot_bar_metric(data, algos5, COL, 'episode_avg_delay', ...
    'End-to-End Delay (ms)', 'Avg Delay (ms)', ...
    fullfile(outputDir, 'fig4_e2e_delay'));

%% ---- Fig 5: Delay Decomposition (stacked) --------------------------------
plot_delay_decomp(data, algos5, COL, DECOMP_COL, ...
    fullfile(outputDir, 'fig5_delay_decomp'));

%% ---- Fig 6: CPU Violation Rate --------------------------------------------
plot_bar_metric(data, algos5, COL, 'episode_cpu_viol_rate', ...
    'CPU Violation Rate', 'Rate (raw / 135)', ...
    fullfile(outputDir, 'fig6_cpu_violation'));

%% ---- Fig 7: Channel Overflow Ratio ----------------------------------------
plot_bar_metric(data, algos5, COL, 'episode_channel_overflow_ratio', ...
    'Channel Overflow Ratio', 'Ratio', ...
    fullfile(outputDir, 'fig7_channel_overflow'));

%% ---- Fig 8: Timeout Ratio --------------------------------------------------
plot_bar_metric(data, algos5, COL, 'episode_timeout_ratio', ...
    'Timeout Ratio', 'Ratio', ...
    fullfile(outputDir, 'fig8_timeout'));

%% ---- Fig 9: Per-Episode Runtime (log scale) --------------------------------
plot_bar_metric(data, algos5, COL, 'episode_runtime_sec', ...
    'Per-Episode Runtime (100 tasks, CPU)', 'Seconds (log scale)', ...
    fullfile(outputDir, 'fig9_runtime'), 'log');

%% ---- Fig 3: multi-seed training reward curves ------------------------------
plot_fig3_seeds(outputDir, COL);

fprintf('\nAll 9 figures saved to %s (.png @300dpi + .svg)\n', outputDir);


%% ============================================================================
%% Local functions
%% ============================================================================

function rgb = hex2rgb(hexStr)
    hexStr = strrep(hexStr, '#', '');
    rgb = [hex2dec(hexStr(1:2)), hex2dec(hexStr(3:4)), hex2dec(hexStr(5:6))] / 255;
end

function savefig_both(fig, basePath)
    % Force a white/light figure regardless of MATLAB's app-level dark-mode
    % theme (R2025a+ figures otherwise inherit a black background on macOS
    % dark mode) so exports match the original matplotlib white-bg figures.
    try
        theme(fig, 'light');
    catch
        % 'theme' function not available pre-R2025a -- explicit colors below
        % already cover that case.
    end
    set(fig, 'Color', [1 1 1], 'InvertHardcopy', 'on');
    axesList = findall(fig, 'Type', 'axes');
    for k = 1:numel(axesList)
        set(axesList(k), 'Color', [1 1 1], 'XColor', [0 0 0], 'YColor', [0 0 0]);
    end
    legendList = findall(fig, 'Type', 'legend');
    for k = 1:numel(legendList)
        set(legendList(k), 'Color', [1 1 1], 'TextColor', [0 0 0]);
    end

    print(fig, [basePath '.png'], '-dpng', '-r300');
    print(fig, [basePath '.svg'], '-dsvg');
    close(fig);
    [~, fname] = fileparts(basePath);
    fprintf('saved %s.png + .svg\n', fname);
end

% ---- generic bar-with-errorbar panel (fig2, fig4, fig6-9) ------------------
function plot_bar_metric(data, algos, COL, key, ttl, ylab, basePath, yscale)
    if nargin < 8
        yscale = 'linear';
    end
    n = numel(algos);
    means  = zeros(1, n);
    stds   = zeros(1, n);
    colors = zeros(n, 3);
    for i = 1:n
        vals = data.(algos{i}).(key);
        means(i) = mean(vals);
        stds(i)  = std(vals);
        colors(i, :) = COL.(algos{i});
    end

    fig = figure('Visible', 'off', 'Position', [100 100 800 500]);
    ax = axes('Parent', fig); hold(ax, 'on');
    b = bar(ax, means, 'FaceColor', 'flat', 'FaceAlpha', 0.85, 'BarWidth', 0.65);
    b.CData = colors;
    errorbar(ax, 1:n, means, stds, 'k', 'LineStyle', 'none', 'LineWidth', 1, 'CapSize', 5);
    for i = 1:n
        text(ax, i, means(i), sprintf('%.3f', means(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
    end
    set(ax, 'XTick', 1:n, 'XTickLabel', algos, 'FontSize', 10);
    if strcmp(yscale, 'log')
        set(ax, 'YScale', 'log');
    end
    title(ax, ttl, 'FontSize', 13, 'FontWeight', 'bold');
    ylabel(ax, ylab, 'FontSize', 10);
    grid(ax, 'on');
    set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.4, 'YGrid', 'on', 'XGrid', 'off');
    box(ax, 'on');

    savefig_both(fig, basePath);
end

% ---- fig5: stacked delay decomposition --------------------------------------
function plot_delay_decomp(data, algos, COL, decompCol, basePath) %#ok<INUSL>
    keys   = {'episode_avg_t_ul', 'episode_avg_t_comp', 'episode_avg_t_link'};
    labels = {'t_{ul} (upload)', 't_{comp} (compute)', 't_{link} (backhaul)'};
    n = numel(algos);
    M = zeros(n, 3);
    for i = 1:n
        for k = 1:3
            M(i, k) = mean(data.(algos{i}).(keys{k}));
        end
    end

    fig = figure('Visible', 'off', 'Position', [100 100 800 500]);
    ax = axes('Parent', fig); hold(ax, 'on');
    b = bar(ax, M, 'stacked', 'FaceAlpha', 0.85, 'BarWidth', 0.65);
    for k = 1:3
        b(k).FaceColor = decompCol(k, :);
    end
    set(ax, 'XTick', 1:n, 'XTickLabel', algos, 'FontSize', 10);
    title(ax, 'Delay Decomposition', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel(ax, 'Delay (ms)', 'FontSize', 10);
    legend(ax, labels, 'Location', 'northeast', 'FontSize', 9);
    grid(ax, 'on');
    set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.4, 'YGrid', 'on', 'XGrid', 'off');
    box(ax, 'on');

    savefig_both(fig, basePath);
end

% ---- fig3: multi-seed training reward curves (SAC/TD3/PPO x orig/seed1/seed2) --
function plot_fig3_seeds(outputDir, COL)
    trainFiles = struct( ...
        'SAC', struct('orig',  'results/sac_v17_training_metrics.json', ...
                       'seed1', 'results/sac_v18_seed1_training_metrics.json', ...
                       'seed2', 'results/sac_v18_seed2_training_metrics.json'), ...
        'TD3', struct('orig',  'results/td3_v17_training_metrics.json', ...
                       'seed1', 'results/td3_v18_seed1_training_metrics.json', ...
                       'seed2', 'results/td3_v18_seed2_training_metrics.json'), ...
        'PPO', struct('orig',  'results/ppo_v17_training_metrics.json', ...
                       'seed1', 'results/ppo_v18_seed1_training_metrics.json', ...
                       'seed2', 'results/ppo_v18_seed2_training_metrics.json'));

    algos       = {'SAC', 'TD3', 'PPO'};
    seedKeys    = {'orig', 'seed1', 'seed2'};
    seedLabels  = {'orig', 'seed 1', 'seed 2'};
    seedStyles  = {'-', '--', ':'};
    seedMarkers = {'none', '^', 'pentagram'};
    seedAlphas  = [0.9, 0.7, 0.7];
    nMarkers    = 22;   % markers per curve -- sparse so ~9900 pts don't clutter
    smoothWin   = 100;

    fig = figure('Visible', 'off', 'Position', [100 100 1200 600]);
    ax = axes('Parent', fig); hold(ax, 'on');

    algoHandles = gobjects(1, numel(algos));
    for a = 1:numel(algos)
        algo = algos{a};
        col  = COL.(algo);
        for s = 1:numel(seedKeys)
            fpath = trainFiles.(algo).(seedKeys{s});
            if ~isfile(fpath)
                warning('Missing %s -- skipping this seed curve', fpath);
                continue;
            end
            d = jsondecode(fileread(fpath));
            y = d.episode_rewards;
            if numel(y) < smoothWin
                continue;
            end
            ySmooth = movmean(y, [smoothWin - 1, 0], 'Endpoints', 'discard');
            x = (smoothWin:numel(y))';
            % offset each seed's marker phase so overlapping curves don't
            % stamp markers on exactly the same x positions
            mStep = max(1, round(numel(x) / nMarkers));
            mStart = 1 + mod((s-1) * round(mStep/3), mStep);
            p = plot(ax, x, ySmooth, seedStyles{s}, 'LineWidth', 1.4, ...
                'Marker', seedMarkers{s}, 'MarkerSize', 6, ...
                'MarkerIndices', mStart:mStep:numel(x));
            p.Color = [col, seedAlphas(s)];
            p.MarkerEdgeColor = col;
        end
        algoHandles(a) = plot(ax, nan, nan, '-', 'Color', col, 'LineWidth', 2);
    end

    seedHandles = gobjects(1, numel(seedKeys));
    for s = 1:numel(seedKeys)
        seedHandles(s) = plot(ax, nan, nan, seedStyles{s}, 'Color', [0.4 0.4 0.4], ...
            'LineWidth', 1.5, 'Marker', seedMarkers{s}, 'MarkerSize', 6, ...
            'MarkerEdgeColor', [0.4 0.4 0.4]);
    end

    xlim(ax, [0 10000]);
    legend(ax, [algoHandles, seedHandles], [algos, seedLabels], ...
        'NumColumns', 3, 'FontSize', 9, 'Location', 'southeast');
    title(ax, 'Training Reward (100-ep moving avg, 3 seeds each)', ...
        'FontSize', 13, 'FontWeight', 'bold');
    xlabel(ax, 'Episode', 'FontSize', 10);
    ylabel(ax, 'Reward', 'FontSize', 10);
    grid(ax, 'on');
    set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.4);
    box(ax, 'on');

    savefig_both(fig, fullfile(outputDir, 'fig3_seeds'));
end

% ---- fig1: system architecture schematic (hand-drawn, no data) -------------
function plot_fig1_system_arch(outputDir)
    black = [0 0 0];
    fig = figure('Visible', 'off', 'Position', [100 100 1600 800]);
    ax = axes('Parent', fig); hold(ax, 'on');
    axis(ax, [0 16 0 8]);
    axis(ax, 'equal');
    axis(ax, 'off');
    set(ax, 'Position', [0.02 0.02 0.96 0.96]);

    chainY = 4.05;

    % ---- 1. IIoT device column -------------------------------------------
    devY = [6.6, 4.9, 3.2, 1.5];
    draw_chip_icon(1.1, devY(1), 0.9);
    draw_arm_icon(1.1, devY(2), 0.9);
    draw_camera_icon(1.1, devY(3), 0.9);
    draw_arm_icon(1.1, devY(4), 0.9);

    % grouping bracket "["
    line(ax, [1.85 1.85], [1.0 7.0], 'Color', black, 'LineWidth', 1.2);
    line(ax, [1.7 1.85], [7.0 7.0], 'Color', black, 'LineWidth', 1.2);
    line(ax, [1.7 1.85], [1.0 1.0], 'Color', black, 'LineWidth', 1.2);
    line(ax, [1.85 2.05], [chainY chainY], 'Color', black, 'LineWidth', 1.2);

    % ---- 2. NOMA uplink / 5G tower ----------------------------------------
    draw_tower_icon(3.6, 4.85, 1.1);
    text(ax, 3.6, 3.55, 'NOMA (Uplink)', 'HorizontalAlignment', 'center', 'FontSize', 12);
    text(ax, 3.6, 3.15, '5G Network', 'HorizontalAlignment', 'center', 'FontSize', 9.5, ...
        'Color', [0.3 0.3 0.3]);
    draw_arrow(ax, 2.05, chainY, 3.05, chainY, black, 1.3, 0.15);
    draw_arrow(ax, 4.3, chainY, 5.3, chainY, black, 1.3, 0.15);

    % ---- 3. Edge Gateway ----------------------------------------------------
    egX = [5.3 7.1]; egY = [3.55 4.55];
    rectangle(ax, 'Position', [egX(1) egY(1) diff(egX) diff(egY)], ...
        'Curvature', [0.08 0.08], 'EdgeColor', black, 'LineWidth', 1.3);
    text(ax, mean(egX), mean(egY), 'Edge Gateway', 'HorizontalAlignment', 'center', 'FontSize', 11);
    draw_wifi_icon(mean(egX), egY(2) + 0.35, 0.5);
    draw_arrow(ax, egX(2), chainY, 7.9, chainY, black, 1.3, 0.15);

    % ---- 4. MEC (server) ------------------------------------------------
    draw_server_icon(8.8, chainY, 1.5, 1.3);
    text(ax, 8.8, 5.0, 'MEC', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    draw_arrow(ax, 9.55, chainY, 10.35, chainY, black, 1.3, 0.15);

    % ---- 5. SFC Processing (VNF chain) -----------------------------------
    vnfCx = [10.9, 12.35, 13.8];
    vnfW = 0.95; vnfH = 0.7;
    draw_vnf_box(vnfCx(1), chainY, vnfW, vnfH, 'VNF1');
    draw_vnf_box(vnfCx(2), chainY, vnfW, vnfH, 'VNF2');
    draw_vnf_box(vnfCx(3), chainY, vnfW, vnfH, 'VNF3');
    line(ax, [vnfCx(1)+vnfW/2, vnfCx(2)-vnfW/2], [chainY chainY], ...
        'Color', black, 'LineStyle', ':', 'LineWidth', 1.3);
    line(ax, [vnfCx(2)+vnfW/2, vnfCx(3)-vnfW/2], [chainY chainY], ...
        'Color', black, 'LineStyle', ':', 'LineWidth', 1.3);
    text(ax, mean(vnfCx), 5.0, 'SFC Processing', 'HorizontalAlignment', 'center', 'FontSize', 11);

    % ---- 6. DRL Manager ----------------------------------------------------
    drlX = [6.7 10.9]; drlY = [0.75 1.85];
    rectangle(ax, 'Position', [drlX(1) drlY(1) diff(drlX) diff(drlY)], ...
        'Curvature', [0.3 0.3], 'EdgeColor', black, 'LineWidth', 1.3);
    text(ax, mean(drlX), mean(drlY)+0.15, 'DRL Manager', ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(ax, mean(drlX), mean(drlY)-0.2, 'Resource & SFC scheduling', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);

    draw_arrow(ax, 7.3, drlY(2), 5.6, 3.7, black, 1.2, 0.15);      % -> Edge Gateway
    draw_arrow(ax, mean(drlX), drlY(2), 8.8, chainY-0.65, black, 1.2, 0.15); % -> MEC
    draw_arrow(ax, 10.3, drlY(2), 10.425, chainY-0.35, black, 1.2, 0.15);   % -> SFC Processing

    savefig_both(fig, fullfile(outputDir, 'fig1_system_arch'));
end

% ---- arrow: shaft (line) + solid triangular head (patch), data coords -----
function draw_arrow(ax, x1, y1, x2, y2, col, lw, hs)
    if nargin < 6, col = [0 0 0]; end
    if nargin < 7, lw = 1.5; end
    if nargin < 8, hs = 0.15; end

    theta = atan2(y2 - y1, x2 - x1);
    xEnd = x2 - hs * cos(theta);
    yEnd = y2 - hs * sin(theta);
    line(ax, [x1 xEnd], [y1 yEnd], 'Color', col, 'LineWidth', lw);

    hw = hs * 0.55;
    px = [x2, xEnd + hw*sin(theta), xEnd - hw*sin(theta)];
    py = [y2, yEnd - hw*cos(theta), yEnd + hw*cos(theta)];
    patch(ax, px, py, col, 'EdgeColor', 'none');
end

% ---- icon: IIoT sensor / chip -------------------------------------------
function draw_chip_icon(cx, cy, s)
    black = [0 0 0];
    rectangle('Position', [cx-s/2, cy-s/2, s, s], 'EdgeColor', black, 'LineWidth', 1.3);
    rectangle('Position', [cx-s/4, cy-s/4, s/2, s/2], 'EdgeColor', black, 'LineWidth', 0.8);
    offs = linspace(-s*0.3, s*0.3, 3);
    for o = offs
        line([cx+o cx+o], [cy-s/2 cy-s/2-s*0.15], 'Color', black, 'LineWidth', 1);
        line([cx+o cx+o], [cy+s/2 cy+s/2+s*0.15], 'Color', black, 'LineWidth', 1);
        line([cx-s/2 cx-s/2-s*0.15], [cy+o cy+o], 'Color', black, 'LineWidth', 1);
        line([cx+s/2 cx+s/2+s*0.15], [cy+o cy+o], 'Color', black, 'LineWidth', 1);
    end
end

% ---- icon: robotic arm ----------------------------------------------------
function draw_arm_icon(cx, cy, s)
    black = [0 0 0];
    baseW = s * 0.7;
    rectangle('Position', [cx-baseW/2, cy-s/2, baseW, s*0.16], ...
        'FaceColor', [0.92 0.92 0.92], 'EdgeColor', black, 'LineWidth', 1.1);
    j0 = [cx, cy - s/2 + s*0.16];
    j1 = [cx - s*0.05, cy + s*0.08];
    j2 = [cx + s*0.35, cy + s*0.38];
    line([j0(1) j1(1)], [j0(2) j1(2)], 'Color', black, 'LineWidth', 2.2);
    line([j1(1) j2(1)], [j1(2) j2(2)], 'Color', black, 'LineWidth', 2.2);
    plot(j0(1), j0(2), 'o', 'MarkerFaceColor', black, 'MarkerEdgeColor', black, 'MarkerSize', 4);
    plot(j1(1), j1(2), 'o', 'MarkerFaceColor', black, 'MarkerEdgeColor', black, 'MarkerSize', 4);
    line([j2(1) j2(1)+s*0.12], [j2(2) j2(2)+s*0.14], 'Color', black, 'LineWidth', 1.6);
    line([j2(1) j2(1)+s*0.16], [j2(2) j2(2)-s*0.02], 'Color', black, 'LineWidth', 1.6);
end

% ---- icon: camera / vision sensor -----------------------------------------
function draw_camera_icon(cx, cy, s)
    black = [0 0 0];
    rectangle('Position', [cx-s*0.45, cy-s*0.28, s*0.9, s*0.5], 'Curvature', [0.3 0.3], ...
        'EdgeColor', black, 'LineWidth', 1.2);
    th = linspace(0, 2*pi, 40);
    r = s * 0.2;
    plot(cx + r*cos(th), cy + 0.02 + r*sin(th), 'Color', black, 'LineWidth', 1.2);
    plot(cx, cy + 0.02, 'o', 'MarkerFaceColor', black, 'MarkerEdgeColor', black, 'MarkerSize', 3);
    line([cx cx], [cy-s*0.28 cy-s*0.48], 'Color', black, 'LineWidth', 1.2);
end

% ---- icon: small wifi arcs --------------------------------------------------
function draw_wifi_icon(cx, cy, s)
    black = [0 0 0];
    plot(cx, cy - s*0.1, 'o', 'MarkerFaceColor', black, 'MarkerEdgeColor', black, 'MarkerSize', 3);
    for r = [0.35 0.6] * s
        th = linspace(deg2rad(30), deg2rad(150), 20);
        plot(cx + r*cos(th), cy - s*0.1 + r*sin(th), 'Color', black, 'LineWidth', 1.0);
    end
end

% ---- icon: 5G tower with signal waves --------------------------------------
function draw_tower_icon(cx, cy, h)
    black = [0 0 0];
    line([cx-h*0.28 cx], [cy-h*0.5 cy+h*0.5], 'Color', black, 'LineWidth', 1.6);
    line([cx+h*0.28 cx], [cy-h*0.5 cy+h*0.5], 'Color', black, 'LineWidth', 1.6);
    line([cx-h*0.14 cx+h*0.14], [cy cy], 'Color', black, 'LineWidth', 1.1);
    line([cx-h*0.06 cx+h*0.06], [cy+h*0.28 cy+h*0.28], 'Color', black, 'LineWidth', 1.1);
    for r = [0.28 0.42 0.56] * h
        th = linspace(deg2rad(25), deg2rad(155), 20);
        plot(cx + r*cos(th), cy + h*0.5 + r*0.6*sin(th), 'Color', black, 'LineWidth', 1.0);
    end
end

% ---- icon: MEC server (3 stacked bars with status LEDs) --------------------
function draw_server_icon(cx, cy, w, h)
    black = [0 0 0];
    nBars = 3;
    barH = h / nBars * 0.78;
    gap  = h / nBars * 0.22;
    y0 = cy - h/2;
    for i = 0:nBars-1
        yb = y0 + i * (barH + gap);
        rectangle('Position', [cx-w/2, yb, w, barH], 'EdgeColor', black, ...
            'FaceColor', [0.95 0.95 0.95], 'LineWidth', 1.2);
        plot(cx - w/2 + w*0.1, yb + barH/2, 'o', 'MarkerFaceColor', [0.2 0.7 0.3], ...
            'MarkerEdgeColor', black, 'MarkerSize', 3);
    end
end

% ---- box: a single VNF stage ------------------------------------------------
function draw_vnf_box(cx, cy, w, h, label)
    black = [0 0 0];
    rectangle('Position', [cx-w/2, cy-h/2, w, h], 'EdgeColor', black, ...
        'FaceColor', 'w', 'LineWidth', 1.3);
    text(cx, cy, label, 'HorizontalAlignment', 'center', 'FontSize', 10);
end
