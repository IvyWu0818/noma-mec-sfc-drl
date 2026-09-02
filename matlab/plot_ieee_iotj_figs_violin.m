% plot_ieee_iotj_figs_violin.m
% Violin-plot companion to the bar charts in plot_ieee_iotj_figs_bigfont.m:
% for every metric that script shows as a mean+errorbar BAR (fig2 reward,
% fig4 delay, fig6 CPU violation, fig7 channel overflow, fig8 timeout,
% fig9 runtime), this makes a SEPARATE figure showing the full distribution
% across the 20 evaluation episodes as a violin (mirrored kernel density
% estimate), with a dash-dot line at the median -- same idea as the
% reference "Miscoverage vs kappa" violin figure, but one violin per
% algorithm instead of per group.
%
% Each violin is filled with the SAME per-algorithm hatch pattern as the
% bar charts (SAC='/', TD3='\', PPO='x', Greedy/GA = same directions at
% higher density -- see HATCH below, identical to plot_ieee_iotj_figs_
% bigfont.m's), for a consistent look across both figure styles. Since a
% violin isn't a rectangle, the hatch lines are swept across the violin's
% bounding box (same reference-height trick as the bars, so the angle/
% spacing is consistent across every violin in one chart) and then trimmed
% down to the violin's actual silhouette by densely sampling each candidate
% segment and keeping only the runs that fall inside the KDE envelope --
% robust to whatever shape the KDE produces (uni/bimodal, skewed, ...),
% unlike a closed-form polygon clip which would need the shape to be convex.
%
% The KDE is hand-rolled (Gaussian kernel, Silverman's rule bandwidth) NOT
% MATLAB's ksdensity -- that needs the Statistics and Machine Learning
% Toolbox, which may not be installed; the hand-rolled version has no
% toolbox dependency at all.
%
% Run from repo root:  matlab -batch "run('matlab/plot_ieee_iotj_figs_violin.m')"
% or open in MATLAB and press Run. Requires R2016b+ (script local functions).

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
cd(repoRoot);

outputDir = fullfile('results', 'figures_paper_violin');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

COL = struct( ...
    'SAC',    hex2rgb('#7E57C2'), ...
    'TD3',    hex2rgb('#2E86AB'), ...
    'PPO',    hex2rgb('#82C882'), ...
    'Greedy', hex2rgb('#E07B54'), ...
    'GA',     hex2rgb('#C2A83E'));

% identical to plot_ieee_iotj_figs_bigfont.m's HATCH, so both figure styles
% read as the same algorithm-to-pattern mapping
HATCH = struct( ...
    'SAC',    struct('style', 'diag1', 'density', 8), ...
    'TD3',    struct('style', 'diag2', 'density', 8), ...
    'PPO',    struct('style', 'cross', 'density', 6), ...
    'Greedy', struct('style', 'diag1', 'density', 18), ...
    'GA',     struct('style', 'cross', 'density', 13));

FS = struct('label', 20, 'tick', 18, 'value', 14, 'showTitles', false);

algos5 = {'SAC', 'TD3', 'PPO', 'Greedy', 'GA'};
algos3 = {'SAC', 'TD3', 'PPO'};

evalFile = fullfile('results', 'final_compare_v20.json');
if ~isfile(evalFile)
    error('Missing %s -- run "python -m experiments.eval_final_compare_v20" first', evalFile);
end
data = jsondecode(fileread(evalFile));

%% ---- Fig 2: Cumulative Reward (DRL only) ---------------------------------
plot_violin_metric(data, algos3, COL, HATCH, FS, 'episode_rewards', ...
    'Reward', fullfile(outputDir, 'fig2_reward_violin'));

%% ---- Fig 4: End-to-End Delay ---------------------------------------------
plot_violin_metric(data, algos5, COL, HATCH, FS, 'episode_avg_delay', ...
    'Avg Delay (ms)', fullfile(outputDir, 'fig4_e2e_delay_violin'));

%% ---- Fig 6: CPU Violation Rate --------------------------------------------
plot_violin_metric(data, algos5, COL, HATCH, FS, 'episode_cpu_viol_rate', ...
    'Rate', fullfile(outputDir, 'fig6_cpu_violation_violin'), 'percent');

%% ---- Fig 7: Channel Overflow Ratio ----------------------------------------
plot_violin_metric(data, algos5, COL, HATCH, FS, 'episode_channel_overflow_ratio', ...
    'Ratio', fullfile(outputDir, 'fig7_channel_overflow_violin'), 'percent');

%% ---- Fig 8: Timeout Ratio --------------------------------------------------
plot_violin_metric(data, algos5, COL, HATCH, FS, 'episode_timeout_ratio', ...
    'Ratio', fullfile(outputDir, 'fig8_timeout_violin'), 'percent');

%% ---- Fig 9: Per-Episode Runtime (log scale) --------------------------------
plot_violin_metric(data, algos5, COL, HATCH, FS, 'episode_runtime_sec', ...
    'Seconds (log scale)', fullfile(outputDir, 'fig9_runtime_violin'), 'log');

fprintf('\nAll violin figures saved to %s (.png @300dpi + .svg)\n', outputDir);


%% ============================================================================
%% Local functions
%% ============================================================================

function rgb = hex2rgb(hexStr)
    hexStr = strrep(hexStr, '#', '');
    rgb = [hex2dec(hexStr(1:2)), hex2dec(hexStr(3:4)), hex2dec(hexStr(5:6))] / 255;
end

function apply_ieee_style(ax, FS)
    set(ax, 'LineWidth', 1.6, 'FontWeight', 'bold', 'FontSize', FS.tick, ...
        'Box', 'on', 'Layer', 'top');
    grid(ax, 'on');
    set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.45, 'GridColor', [0.45 0.45 0.45], ...
        'YGrid', 'on', 'XGrid', 'off');
end

function savefig_both(fig, basePath)
    try
        theme(fig, 'light');
    catch
    end
    set(fig, 'Color', [1 1 1]);
    axesList = findall(fig, 'Type', 'axes');
    for k = 1:numel(axesList)
        set(axesList(k), 'Color', [1 1 1], 'XColor', [0 0 0], 'YColor', [0 0 0]);
    end
    drawnow;
    tighten_axes(fig);
    print(fig, [basePath '.png'], '-dpng', '-r300');
    print(fig, [basePath '.svg'], '-dsvg');
    close(fig);
    [~, fname] = fileparts(basePath);
    fprintf('saved %s.png + .svg\n', fname);
end

function tighten_axes(fig)
    axesList = findall(fig, 'Type', 'axes');
    for k = 1:numel(axesList)
        ax = axesList(k);
        outerpos = ax.OuterPosition;
        ti = ax.TightInset;
        pad = 0.01;
        left   = outerpos(1) + ti(1) + pad;
        bottom = outerpos(2) + ti(2) + pad;
        w = max(outerpos(3) - ti(1) - ti(3) - 2*pad, 0.05);
        h = max(outerpos(4) - ti(2) - ti(4) - 2*pad, 0.05);
        ax.Position = [left bottom w h];
    end
end

% ---- Gaussian KDE, Silverman's rule bandwidth. No toolbox dependency
% (unlike ksdensity, which needs Statistics and Machine Learning Toolbox).--
function [f, xi] = gaussian_kde(vals, nPoints, padFactor)
    vals = vals(:);
    n = numel(vals);
    sigma = std(vals);
    if sigma < eps
        sigma = max(abs(vals)) * 0.05 + eps;
    end
    bw = max(1.06 * sigma * n^(-1/5), eps);
    lo = min(vals) - padFactor * bw;
    hi = max(vals) + padFactor * bw;
    xi = linspace(lo, hi, nPoints)';
    f = zeros(size(xi));
    for k = 1:n
        f = f + exp(-0.5 * ((xi - vals(k)) / bw) .^ 2);
    end
    f = f / (n * bw * sqrt(2 * pi));
end

% ---- Liang-Barsky clip of the infinite line P(t) = (px,py)+t*(dx,dy)
% against the box [xmin,xmax] x [ymin,ymax]. t0/t1 start at -Inf/+Inf --
% (px,py) is just some point ON the line, not necessarily where it enters
% the box, so the box-intersecting portion can be on either side of it. ----
function [t0, t1, valid] = liang_barsky_clip(px, py, dx, dy, xmin, xmax, ymin, ymax)
    t0 = -Inf; t1 = Inf; valid = true;
    p = [-dx, dx, -dy, dy];
    q = [px - xmin, xmax - px, py - ymin, ymax - py];
    for i = 1:4
        if p(i) == 0
            if q(i) < 0
                valid = false;
                return;
            end
        else
            r = q(i) / p(i);
            if p(i) < 0
                if r > t1
                    valid = false;
                    return;
                elseif r > t0
                    t0 = r;
                end
            else
                if r < t0
                    valid = false;
                    return;
                elseif r < t1
                    t1 = r;
                end
            end
        end
    end
end

% ---- one family of parallel diagonal lines, swept across THIS violin's
% bounding box [xc-maxHW, xc+maxHW] x [y0, yTop] using a slope/spacing
% derived from the CHART-WIDE reference (refW, refH) so every violin in one
% figure has the identical real angle and line spacing (same idea as
% plot_ieee_iotj_figs_bigfont.m's bar hatch). Each rectangle-clipped
% candidate line is then trimmed down to the violin's actual silhouette in
% trim_to_violin. dir=+1 is '/', dir=-1 is '\'. ------------------------------
function sweep_violin_diag(ax, xc, xiLocal, halfW, dir, density, refW, refH, color, lw)
    y0 = xiLocal(1);
    thisH = xiLocal(end) - xiLocal(1);
    thisW = 2 * max(halfW);
    if thisH <= 0 || thisW <= 0
        return;
    end

    slope = refH / refW;
    dx = 1; dy = dir * slope;
    nlen = hypot(dx, dy);
    nxu = -dy / nlen; nyu = dx / nlen;

    refCorners = [0 0; refW 0; refW refH; 0 refH];
    refProj = refCorners * [nxu; nyu];
    spacing = (max(refProj) - min(refProj)) / (2 * density);

    corners = [0 0; thisW 0; thisW thisH; 0 thisH];
    proj = corners * [nxu; nyu];
    offMin = min(proj); offMax = max(proj);
    nLines = max(1, round((offMax - offMin) / spacing));

    for i = 0:nLines
        off = offMin + i * (offMax - offMin) / nLines;
        basePt = off * [nxu, nyu];
        [t0, t1, valid] = liang_barsky_clip(basePt(1), basePt(2), dx, dy, 0, thisW, 0, thisH);
        if ~valid || t1 <= t0
            continue;
        end
        p1 = basePt + t0 * [dx, dy];
        p2 = basePt + t1 * [dx, dy];
        % p1/p2 are in this violin's LOCAL box coords: x in [0,thisW]
        % (offset from the box's left edge), y in [0,thisH] (offset from y0)
        trim_to_violin(ax, xc, thisW, y0, p1, p2, xiLocal, halfW, color, lw);
    end
end

% ---- densely sample the candidate segment [p1,p2] (local box coords) and
% keep only the contiguous runs that actually fall inside the violin's
% envelope |x-xc| <= halfW(y) -- correct regardless of how irregular the
% KDE silhouette is (no assumption that the violin is convex). -------------
function trim_to_violin(ax, xc, thisW, y0, p1, p2, xiLocal, halfW, color, lw)
    nSamp = 60;
    t = linspace(0, 1, nSamp);
    localX = p1(1) + t * (p2(1) - p1(1));
    localY = p1(2) + t * (p2(2) - p1(2));
    actualX = xc - thisW / 2 + localX;
    actualY = y0 + localY;
    hw = interp1(xiLocal, halfW, actualY, 'linear', 0);
    inside = abs(actualX - xc) <= hw;

    d = diff([false, inside, false]);
    starts = find(d == 1);
    stops  = find(d == -1) - 1;
    for k = 1:numel(starts)
        idx = starts(k):stops(k);
        if numel(idx) >= 2
            ln = line(ax, actualX(idx), actualY(idx), 'LineWidth', lw);
            ln.Color = color;
        end
    end
end

% ---- hatch dispatcher for one violin, mirroring draw_hatch_bar's
% style/density -> direction(s) mapping in the bigfont bar charts. ---------
function draw_violin_hatch(ax, xc, xiLocal, halfW, hatchCfg, refW, refH)
    color = [0 0 0 0.4];
    lw = 1.0;
    switch hatchCfg.style
        case 'diag1'
            sweep_violin_diag(ax, xc, xiLocal, halfW, +1, hatchCfg.density, refW, refH, color, lw);
        case 'diag2'
            sweep_violin_diag(ax, xc, xiLocal, halfW, -1, hatchCfg.density, refW, refH, color, lw);
        case 'cross'
            sweep_violin_diag(ax, xc, xiLocal, halfW, +1, hatchCfg.density, refW, refH, color, lw);
            sweep_violin_diag(ax, xc, xiLocal, halfW, -1, hatchCfg.density, refW, refH, color, lw);
    end
end

% ---- one violin: mirrored KDE filled patch, hatch pattern, and a dash-dot
% line at the median (matching the reference figure's style). `xiLocal`
% passed in is ALREADY in whatever local coordinate system the hatch sweep
% should use (log10-local for a log-scaled axis, matching
% plot_ieee_iotj_figs_bigfont.m's bar hatch fix -- see the caller). ---------
function draw_violin(ax, xc, xi, xiLocal, f, medVal, medLocal, faceColor, maxHalfWidth, hatchCfg, refW, refH)
    if max(f) > 0
        halfW = f / max(f) * maxHalfWidth;
    else
        halfW = zeros(size(f));
    end

    patch(ax, [xc - halfW; flipud(xc + halfW)], [xi; flipud(xi)], faceColor, ...
        'FaceAlpha', 0.65, 'EdgeColor', faceColor * 0.5, 'LineWidth', 1.3);

    draw_violin_hatch(ax, xc, xiLocal, halfW, hatchCfg, refW, refH);

    medHalfW = interp1(xiLocal, halfW, medLocal, 'linear', 'extrap');
    medHalfW = max(medHalfW, 0.02 * maxHalfWidth);
    ln = line(ax, [xc - medHalfW, xc + medHalfW], [medVal, medVal], ...
        'LineStyle', '-.', 'LineWidth', 2.2);
    ln.Color = faceColor * 0.45;
end

% ---- generic violin panel: one violin per algo, at x = 1..n, x-ticks
% labeled by algo name (no legend needed -- unlike the reference figure,
% algorithm identity is already the x-axis here, not a separate grouping).
% yscale: 'linear' (default), 'log' (fig9), or 'percent' (fig6-8). KDEs are
% computed in a first pass (needed to find refH -- the largest violin's
% local-coordinate range in this chart -- before any hatch can be drawn
% with a chart-consistent angle/spacing), then everything is drawn. --------
function plot_violin_metric(data, algos, COL, HATCH, FS, key, ylab, basePath, yscale)
    if nargin < 9
        yscale = 'linear';
    end
    n = numel(algos);
    isLog = strcmp(yscale, 'log');
    isPercent = strcmp(yscale, 'percent');
    maxHalfWidth = 0.38;

    fig = figure('Visible', 'off', 'Position', [100 100 950 620]);
    ax = axes('Parent', fig); hold(ax, 'on');

    % ---- pass 1: KDE per algo + chart-wide reference box ------------------
    kde = struct('xi', {}, 'xiLocal', {}, 'f', {}, 'medVal', {}, 'medLocal', {}, 'meanVal', {});
    refW = 2 * maxHalfWidth;
    refH = 0;
    for i = 1:n
        vals = data.(algos{i}).(key);
        if isPercent
            vals = vals * 100;
        end
        if isLog
            [f, xiLocal] = gaussian_kde(log10(vals), 200, 3);
            xi = 10 .^ xiLocal;
            medVal = median(vals);
            medLocal = log10(medVal);
        else
            [f, xi] = gaussian_kde(vals, 200, 3);
            xiLocal = xi;
            medVal = median(vals);
            medLocal = medVal;
        end
        kde(i) = struct('xi', xi, 'xiLocal', xiLocal, 'f', f, ...
            'medVal', medVal, 'medLocal', medLocal, 'meanVal', mean(vals)); %#ok<AGROW>
        refH = max(refH, xiLocal(end) - xiLocal(1));
    end

    % ---- pass 2: draw ----------------------------------------------------
    % mean-value label placed just above the violin's own top edge, same
    % "va=bottom right at the anchor point" convention as the bar charts'
    % value labels (which sit just above the error-bar cap). -----------------
    for i = 1:n
        draw_violin(ax, i, kde(i).xi, kde(i).xiLocal, kde(i).f, kde(i).medVal, ...
            kde(i).medLocal, COL.(algos{i}), maxHalfWidth, HATCH.(algos{i}), refW, refH);
        text(ax, i, max(kde(i).xi), sprintf('%.3f', kde(i).meanVal), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', FS.value, 'FontWeight', 'bold');
    end

    set(ax, 'XTick', 1:n, 'XTickLabel', algos);
    xlim(ax, [0.5, n + 0.5]);
    if isLog
        set(ax, 'YScale', 'log');
    end
    ylabel(ax, ylab, 'FontSize', FS.label, 'FontWeight', 'bold');
    apply_ieee_style(ax, FS);

    savefig_both(fig, basePath);
end
