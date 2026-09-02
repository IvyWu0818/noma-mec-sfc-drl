% plot_ieee_iotj_figs_bigfont.m
% "IEEE conference paper" style variant of fig2_reward ... fig9_runtime:
% same data sources and same per-algorithm color palette as
% matlab/plot_ieee_iotj_figs.m, but restyled to look like a typeset MATLAB
% figure rather than a matplotlib one -- thick bold axis box, bold ticks/
% labels, percent-formatted ratio axes, top horizontal legends where a
% legend is needed, no per-bar numeric callouts, and (for the combined
% fig4-fig9 panel) sub-captions "(a) ..." placed BELOW each tile instead of
% as a title above it. fig3 additionally gets a shaded uncertainty band
% around each algorithm's main training curve.
%
% Hatch patterns are hand-drawn (patch/line), NOT via the third-party
% hatchfill2 -- that was tried and reverted: its 'speckle' style is a
% documented known-broken bug in this release, and its live axis-tracking
% listeners corrupt/misalign the hatch geometry when many bar()+hatchfill2
% calls stack up in one axes (our delay-decomposition and combined-eval
% panels call it up to 15x per figure). The hand-rolled version below has
% no such dependency and is deterministic.
%
% Per-algorithm hatch patterns (style + density, see HATCH struct below):
% SAC=diag1 '/', TD3=diag2 '\', PPO=cross 'x' (all medium density);
% Greedy=diag1 '/' at high density (fine hatching), GA=cross 'x' at high
% density (fine grid) -- same two directions as SAC/PPO, differentiated by
% line spacing instead of angle. An angle-("slope"-)adjustable version of
% draw_diag was tried and re-broke three times in a row (bad x-only
% clipping, an un-normalized sweep vector, uneven cross spacing), so this
% sticks to the one geometry that's actually been reliable throughout.
%
% fig1_system_arch (the hand-drawn schematic) is unchanged -- it has no bars
% or data-driven text sizing issue, so it isn't reproduced here. Run
% plot_ieee_iotj_figs.m if you need that one.
%
% Run from repo root:  matlab -batch "run('matlab/plot_ieee_iotj_figs_bigfont.m')"
% or open in MATLAB and press Run. Requires R2016b+ (script local functions).

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
cd(repoRoot);

outputDir = fullfile('results', 'figures_paper_bigfont');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% ---- shared color palette (identical to plot_ieee_iotj_figs.m) ----
COL = struct( ...
    'SAC',    hex2rgb('#7E57C2'), ...
    'TD3',    hex2rgb('#2E86AB'), ...
    'PPO',    hex2rgb('#82C882'), ...
    'Greedy', hex2rgb('#E07B54'), ...
    'GA',     hex2rgb('#C2A83E'));

DECOMP_COL = [hex2rgb('#7DC3E8'); hex2rgb('#F0A070'); hex2rgb('#82C882')];

% ---- one hatch pattern per algorithm (style + line density); drawn in a
% semi-transparent black on top of the algorithm's own fill color, so color
% distinguishes at a glance and pattern still works in grayscale/print -----
% NOTE: differentiated by style + density only (no angle/"slope" trick) --
% that was tried 3 times and kept re-breaking (bad clipping, un-normalized
% sweep vector, uneven cross spacing). This is the exact geometry that was
% already proven correct for SAC/TD3/PPO across many rounds; Greedy/GA
% reuse the same two directions at very different densities (coarse '/' vs
% fine '/', coarse 'x' vs fine 'x' grid) so they still read as distinct.
HATCH = struct( ...
    'SAC',    struct('style', 'diag1', 'density', 8), ...    % '/', medium
    'TD3',    struct('style', 'diag2', 'density', 8), ...    % '\', medium
    'PPO',    struct('style', 'cross', 'density', 6), ...    % 'x', medium grid
    'Greedy', struct('style', 'diag1', 'density', 18), ...   % '/', fine/dense
    'GA',     struct('style', 'cross', 'density', 13));      % 'x', fine/dense grid

% ---- font sizes: bump every text element up from the standard figs ----
% showTitles=false: the paper already captions each figure, so the in-image
% chart titles are redundant -- toggle back to true to bring them back.
% (a)/(b)/(c) sub-panel letters in the combined figure are unaffected, those
% identify panels within one multi-panel figure rather than duplicate a
% caption. -------------------------------------------------------------------
FS = struct('title', 24, 'label', 20, 'tick', 18, 'value', 14, 'legend', 13, ...
    'showTitles', false);

algos5 = {'SAC', 'TD3', 'PPO', 'Greedy', 'GA'};
algos3 = {'SAC', 'TD3', 'PPO'};

evalFile = fullfile('results', 'final_compare_v20.json');
if ~isfile(evalFile)
    error('Missing %s -- run "python -m experiments.eval_final_compare_v20" first', evalFile);
end
data = jsondecode(fileread(evalFile));

%% ---- Fig 2 (old): Cumulative Reward bar chart (DRL only) -----------------
plot_bar_metric(data, algos3, COL, HATCH, FS, 'episode_rewards', ...
    'Cumulative Reward (DRL only)', 'Reward', ...
    fullfile(outputDir, 'fig2_reward_bar'));

%% ---- Fig 2 (new): Episode reward vs training episode (DRL only) ----------
plot_fig2_reward_curve(outputDir, COL, FS, algos3, data);

%% ---- Fig 4: End-to-End Delay ---------------------------------------------
plot_bar_metric(data, algos5, COL, HATCH, FS, 'episode_avg_delay', ...
    'End-to-End Delay (ms)', 'Avg Delay (ms)', ...
    fullfile(outputDir, 'fig4_e2e_delay'));

%% ---- Fig 5: Delay Decomposition (stacked) --------------------------------
plot_delay_decomp(data, algos5, COL, DECOMP_COL, HATCH, FS, ...
    fullfile(outputDir, 'fig5_delay_decomp'));

%% ---- Fig 6: CPU Violation Rate --------------------------------------------
plot_bar_metric(data, algos5, COL, HATCH, FS, 'episode_cpu_viol_rate', ...
    'CPU Violation Rate', 'Rate', ...
    fullfile(outputDir, 'fig6_cpu_violation'), 'percent');

%% ---- Fig 7: Channel Overflow Ratio ----------------------------------------
plot_bar_metric(data, algos5, COL, HATCH, FS, 'episode_channel_overflow_ratio', ...
    'Channel Overflow Ratio', 'Ratio', ...
    fullfile(outputDir, 'fig7_channel_overflow'), 'percent');

%% ---- Fig 8: Timeout Ratio --------------------------------------------------
plot_bar_metric(data, algos5, COL, HATCH, FS, 'episode_timeout_ratio', ...
    'Timeout Ratio', 'Ratio', ...
    fullfile(outputDir, 'fig8_timeout'), 'percent');

%% ---- Fig 9: Per-Episode Runtime (log scale) --------------------------------
plot_bar_metric(data, algos5, COL, HATCH, FS, 'episode_runtime_sec', ...
    'Per-Episode Runtime (100 tasks, CPU)', 'Seconds (log scale)', ...
    fullfile(outputDir, 'fig9_runtime'), 'log');

%% ---- Fig 3: multi-seed training reward curves ------------------------------
plot_fig3_seeds(outputDir, COL, FS);

%% ---- Combined fig4-fig9: one 2x3 multi-panel figure, (a)-(f) sub-labels ---
plot_combined_eval(data, algos5, COL, DECOMP_COL, HATCH, FS, ...
    fullfile(outputDir, 'fig_combined_eval'));

fprintf('\nAll figures saved to %s (.png @300dpi + .svg)\n', outputDir);


%% ============================================================================
%% Local functions
%% ============================================================================

function rgb = hex2rgb(hexStr)
    hexStr = strrep(hexStr, '#', '');
    rgb = [hex2dec(hexStr(1:2)), hex2dec(hexStr(3:4)), hex2dec(hexStr(5:6))] / 255;
end

% ---- "typeset paper" look: thick black axis box, bold ticks/labels, dashed
% gray horizontal-only grid. Applied to every axes right before saving. -----
function apply_ieee_style(ax, FS)
    set(ax, 'LineWidth', 1.6, 'FontWeight', 'bold', 'FontSize', FS.tick, ...
        'Box', 'on', 'Layer', 'top');
    grid(ax, 'on');
    set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.45, 'GridColor', [0.45 0.45 0.45], ...
        'YGrid', 'on', 'XGrid', 'off');
end

function savefig_both(fig, basePath, skipTighten)
    if nargin < 3
        skipTighten = false;   % set true if the caller already tightened the
    end                        % layout itself (e.g. to measure a legend's
                                % final position before adding something next to it)
    % Force a white/light figure regardless of MATLAB's app-level dark-mode
    % theme (R2025a+ figures otherwise inherit a black background on macOS
    % dark mode) so exports match the original matplotlib white-bg figures.
    try
        theme(fig, 'light');
    catch
        % 'theme' function not available pre-R2025a -- explicit colors below
        % already cover that case.
    end
    set(fig, 'Color', [1 1 1]);
    axesList = findall(fig, 'Type', 'axes');
    for k = 1:numel(axesList)
        set(axesList(k), 'Color', [1 1 1], 'XColor', [0 0 0], 'YColor', [0 0 0]);
    end
    legendList = findall(fig, 'Type', 'legend');
    for k = 1:numel(legendList)
        set(legendList(k), 'Color', [1 1 1], 'TextColor', [0 0 0], ...
            'EdgeColor', [0 0 0], 'FontWeight', 'bold');
    end

    drawnow;             % force a layout pass so TightInset below is accurate
    if ~skipTighten
        tighten_axes(fig);   % trim left/right/top/bottom whitespace around single axes
    end

    print(fig, [basePath '.png'], '-dpng', '-r300');
    print(fig, [basePath '.svg'], '-dsvg');
    close(fig);
    [~, fname] = fileparts(basePath);
    fprintf('saved %s.png + .svg\n', fname);
end

% ---- shrink each axes' Position down to its TightInset (kills the wide
% left/right margins single-axes figures get by default). Tiledlayout tiles
% are skipped -- 'TileSpacing'/'Padding' already control their spacing.
% Axes with an OUTSIDE-positioned legend (e.g. 'northoutside') are also
% skipped: TightInset doesn't know about that legend's reserved space, so
% forcing this axes to the "tight" size would grow it back over the legend.
% MATLAB's own automatic outside-legend layout is left in charge there. -----
function tighten_axes(fig)
    axesList = findall(fig, 'Type', 'axes');
    for k = 1:numel(axesList)
        ax = axesList(k);
        if isa(ax.Parent, 'matlab.graphics.layout.TiledChartLayout')
            continue;
        end
        lg = get(ax, 'Legend');
        if ~isempty(lg) && isgraphics(lg) && contains(lower(lg.Location), 'outside')
            continue;
        end
        outerpos = ax.OuterPosition;
        ti = ax.TightInset;
        pad = 0.01;  % tiny breathing room so tick labels don't touch the edge
        left   = outerpos(1) + ti(1) + pad;
        bottom = outerpos(2) + ti(2) + pad;
        w = max(outerpos(3) - ti(1) - ti(3) - 2*pad, 0.05);
        h = max(outerpos(4) - ti(2) - ti(4) - 2*pad, 0.05);
        ax.Position = [left bottom w h];
    end
end

% ---- draw one hatched, colored bar (rectangle) at bar-index `xc`. Pass
% logY=true when the bar sits on a log-scaled y-axis (e.g. fig9 runtime) so
% the hatch geometry is interpolated in log-space -- interpolating in linear
% space there made the pattern look warped/uneven once the log axis
% stretched the bottom of the bar and compressed the top.
%
% `refH` is the tallest bar's height IN THIS CHART (pass the same value for
% every bar drawn in one figure). Without it, each bar's hatch angle is its
% OWN corner-to-corner diagonal (h_i/w), which makes a tall bar's hatch
% look denser/steeper than a short bar's -- purely a side effect of their
% different aspect ratios, not a deliberate style choice. Sharing one refH
% across all bars gives every one of them the same real slope and the same
% real line spacing; a shorter bar just shows fewer repeats of that same
% pattern instead of a squished one. ----------------------------------------
function draw_one_bar(ax, xc, yBottom, yTop, barWidth, faceColor, hatchStyle, hatchDensity, logY, refH)
    if nargin < 9
        logY = false;
    end
    if nargin < 10
        refH = [];
    end
    x0 = xc - barWidth/2;
    w  = barWidth;
    y0 = min(yBottom, yTop);
    h  = abs(yTop - yBottom);
    patch(ax, [x0 x0+w x0+w x0], [y0 y0 y0+h y0+h], faceColor, ...
        'FaceAlpha', 0.85, 'EdgeColor', faceColor * 0.55, 'LineWidth', 1.2);
    if h > 1e-9
        draw_hatch_bar(ax, x0, y0, w, h, hatchStyle, [0 0 0 0.45], 1.0, hatchDensity, logY, refH);
    end
end

% ---- hatch dispatcher: draws a pattern of diagonal lines clipped to the
% given bar rectangle [x0,y0,w,h]. See draw_diag for what `refH` does. -----
function draw_hatch_bar(ax, x0, y0, w, h, style, color, lw, density, logY, refH)
    if nargin < 10
        logY = false;
    end
    if nargin < 11 || isempty(refH)
        refH = h;
    end
    switch style
        case 'diag1'   % '/'
            draw_diag(ax, x0, y0, w, h, +1, color, lw, density, logY, refH);
        case 'diag2'   % '\'
            draw_diag(ax, x0, y0, w, h, -1, color, lw, density, logY, refH);
        case 'cross'   % 'x'
            draw_diag(ax, x0, y0, w, h, +1, color, lw, density, logY, refH);
            draw_diag(ax, x0, y0, w, h, -1, color, lw, density, logY, refH);
    end
end

% ---- one family of parallel diagonal lines filling [x0,y0,w,h]. dir=+1 is
% '/' (rises left-to-right), dir=-1 is '\' (falls left-to-right). The line
% direction and inter-line spacing are both derived from (w, refH) -- i.e.
% from a REFERENCE box shared by every bar in the chart, not from this
% particular bar's own (w,h) -- so every bar's hatch has the identical real
% slope and spacing; only the NUMBER of repeats varies with this bar's own
% height. Lines are found via a standard Liang-Barsky clip of the swept
% line family against a LOCAL box (checks all 4 box edges at once -- an
% earlier x-only-clip version silently ignored the y bound and broke as
% soon as the line's slope didn't already match h/w exactly).
%
% When logY is set (fig9's log-scaled runtime axis), that local box and all
% the slope/spacing/clip math are done in LOG-Y LOCAL COORDINATES
% (localY = log10(y0+dataY) - log10(y0)) instead of raw data. An earlier
% version only log-transformed the two final endpoints of each already-
% linear-space line; that keeps each individual segment straight, but the
% SPACING between successive swept lines still came from linear space, so
% it got compressed/stretched unevenly once displayed on the log axis --
% a PPO 'cross' hatch that looks like a clean grid on every linear chart
% collapsed toward looking like a single dominant direction on fig9.
% Working entirely in log-local coordinates and exponentiating back only at
% the very end keeps the same real grid shape on a log axis too. -----------
function draw_diag(ax, x0, y0, w, h, dir, color, lw, density, logY, refH)
    if nargin < 10
        logY = false;
    end
    if nargin < 11 || isempty(refH)
        refH = h;
    end

    if logY
        hLocal    = log10(y0 + h) - log10(y0);
        refHLocal = log10(y0 + refH) - log10(y0);
    else
        hLocal    = h;
        refHLocal = refH;
    end

    slope = refHLocal / w;
    dx = 1; dy = dir * slope;
    nlen = hypot(dx, dy);
    nxu = -dy / nlen; nyu = dx / nlen;   % unit normal, shared sweep direction

    % spacing fixed by the REFERENCE box, so it's identical for every bar
    refCorners = [0 0; w 0; w refHLocal; 0 refHLocal];
    refProj = refCorners * [nxu; nyu];
    spacing = (max(refProj) - min(refProj)) / (2 * density);

    % how many of those fixed-spacing lines actually cross THIS bar's box
    corners = [0 0; w 0; w hLocal; 0 hLocal];
    proj = corners * [nxu; nyu];
    offMin = min(proj); offMax = max(proj);
    nLines = max(1, round((offMax - offMin) / spacing));

    for i = 0:nLines
        off = offMin + i * (offMax - offMin) / nLines;
        basePt = off * [nxu, nyu];
        [t0, t1, valid] = liang_barsky_clip(basePt(1), basePt(2), dx, dy, 0, w, 0, hLocal);
        if ~valid || t1 <= t0
            continue;
        end
        p1 = basePt + t0 * [dx, dy];
        p2 = basePt + t1 * [dx, dy];
        xa = x0 + p1(1);
        xb = x0 + p2(1);
        if logY
            ya = 10 ^ (log10(y0) + p1(2));
            yb = 10 ^ (log10(y0) + p2(2));
        else
            ya = y0 + p1(2);
            yb = y0 + p2(2);
        end
        ln = line(ax, [xa xb], [ya yb], 'LineWidth', lw);
        ln.Color = color;
    end
end

% ---- Liang-Barsky clip of the parametric segment P(t) = (px,py)+t*(dx,dy),
% t in [0,1], against the box [xmin,xmax] x [ymin,ymax]. Standard textbook
% algorithm -- checks all 4 box edges together, unlike clipping x and y
% separately. Returns the surviving t-range and whether it's non-empty. ----
function [t0, t1, valid] = liang_barsky_clip(px, py, dx, dy, xmin, xmax, ymin, ymax)
    % t0/t1 start at -Inf/+Inf -- (px,py) is just SOME point ON the infinite
    % line, not necessarily where it enters the box, so the box-intersecting
    % portion can just as easily be on the t<0 side as on t>0. Starting from
    % [0,1] here (an earlier version's bug) silently discarded any line whose
    % basePt happened to sit past where the box's intersection begins,
    % which is what made hatch density collapse toward one direction only
    % (most visible in fig9's steep log-space slope, but present whenever a
    % swept basePt landed past the box edge in the forward direction).
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

% ---- generic hatched-bar panel (fig2, fig4, fig6-9). yscale: 'linear'
% (default), 'log' (fig9), or 'percent' (ratio metrics -- the underlying
% 0-1 fraction is scaled by 100 up front, e.g. 0.062 -> 6.2, and plotted as
% a plain number; no '%' suffix on the axis or the bar labels). -------------
function plot_bar_metric(data, algos, COL, HATCH, FS, key, ttl, ylab, basePath, yscale)
    if nargin < 10
        yscale = 'linear';
    end
    isPercent = strcmp(yscale, 'percent');
    n = numel(algos);
    means = zeros(1, n);
    stds  = zeros(1, n);
    for i = 1:n
        vals = data.(algos{i}).(key);
        if isPercent
            vals = vals * 100;
        end
        means(i) = mean(vals);
        stds(i)  = std(vals);
    end

    fig = figure('Visible', 'off', 'Position', [100 100 950 580]);
    ax = axes('Parent', fig); hold(ax, 'on');

    barWidth = 0.65;
    isLog = strcmp(yscale, 'log');
    if isLog
        yBase = min(means) / 3;   % bars "start" here instead of 0 (log(0) undefined)
    else
        yBase = 0;
    end
    % tallest bar's height -- shared hatch reference (see draw_diag). abs()
    % matters for fig2's reward bars, which are all negative: each bar's
    % own drawn height is abs(means(i)-yBase), so the reference must match.
    refH = max(abs(means - yBase));

    for i = 1:n
        h = HATCH.(algos{i});
        if isLog
            draw_one_bar(ax, i, yBase, means(i), barWidth, COL.(algos{i}), h.style, h.density, true, refH);
        else
            draw_one_bar(ax, i, 0, means(i), barWidth, COL.(algos{i}), h.style, h.density, false, refH);
        end
    end

    errorbar(ax, 1:n, means, stds, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);

    for i = 1:n
        labelStr = sprintf('%.3f', means(i));
        if means(i) >= 0
            text(ax, i, means(i) + stds(i), labelStr, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', FS.value, 'FontWeight', 'bold');
        else
            text(ax, i, means(i) - stds(i), labelStr, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'top', 'FontSize', FS.value, 'FontWeight', 'bold');
        end
    end

    set(ax, 'XTick', 1:n, 'XTickLabel', algos);
    xlim(ax, [0.5, n + 0.5]);
    if isLog
        set(ax, 'YScale', 'log');
        ylim(ax, [yBase, max(means + stds) * 2]);
    else
        if min(means) >= 0
            % Bars here can never be negative (delay/ratio/etc.) -- pin the
            % bottom to 0 so MATLAB's auto-tick padding doesn't sneak a
            % negative range in below data that's always >= 0. Reward
            % charts (means < 0) are untouched and keep their auto range.
            yl = ylim(ax);
            ylim(ax, [0, yl(2)]);
        end
    end
    if FS.showTitles
        title(ax, ttl, 'FontSize', FS.title, 'FontWeight', 'bold');
    end
    ylabel(ax, ylab, 'FontSize', FS.label, 'FontWeight', 'bold');
    apply_ieee_style(ax, FS);

    savefig_both(fig, basePath);
end

% ---- fig5: stacked delay decomposition, hatched per-algorithm, top legend --
function plot_delay_decomp(data, algos, COL, decompCol, HATCH, FS, basePath) %#ok<INUSL>
    keys   = {'episode_avg_t_ul', 'episode_avg_t_comp', 'episode_avg_t_link'};
    labels = {'t_{ul} (upload)', 't_{comp} (compute)', 't_{link} (backhaul)'};
    n = numel(algos);
    M = zeros(n, 3);
    for i = 1:n
        for k = 1:3
            M(i, k) = mean(data.(algos{i}).(keys{k}));
        end
    end

    fig = figure('Visible', 'off', 'Position', [100 100 950 620]);
    ax = axes('Parent', fig); hold(ax, 'on');

    barWidth = 0.65;
    refH = max(M(:));   % largest single segment in the chart -- shared hatch reference
    legHandles = gobjects(1, 3);
    totals = sum(M, 2);
    for i = 1:n
        h = HATCH.(algos{i});
        yBottom = 0;
        for k = 1:3
            seg = M(i, k);
            p = patch(ax, ...
                [i-barWidth/2, i+barWidth/2, i+barWidth/2, i-barWidth/2], ...
                [yBottom, yBottom, yBottom+seg, yBottom+seg], ...
                decompCol(k, :), 'FaceAlpha', 0.85, 'EdgeColor', decompCol(k, :) * 0.55, 'LineWidth', 1.2);
            if i == 1
                legHandles(k) = p;
            end
            if seg > 1e-9
                draw_hatch_bar(ax, i-barWidth/2, yBottom, barWidth, seg, ...
                    h.style, [0 0 0 0.45], 1.0, h.density, false, refH);
            end
            yBottom = yBottom + seg;
        end
        text(ax, i, totals(i), sprintf('%.3f', totals(i)), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontSize', FS.value, 'FontWeight', 'bold');
    end

    set(ax, 'XTick', 1:n, 'XTickLabel', algos);
    xlim(ax, [0.5, n + 0.5]);
    % Delay is never negative -- pin the bottom to 0. Also add extra
    % headroom above the tallest bar (computed directly off the tallest
    % stacked total, not MATLAB's auto-padded axis limit, to avoid
    % double-padding) so the inside-top legend has empty space to sit in
    % instead of overlapping the tallest bar's top segment + its label.
    ylim(ax, [0, max(totals) * 1.22]);
    if FS.showTitles
        title(ax, 'Delay Decomposition', 'FontSize', FS.title, 'FontWeight', 'bold');
    end
    ylabel(ax, 'Delay (ms)', 'FontSize', FS.label, 'FontWeight', 'bold');
    legend(ax, legHandles, labels, 'Location', 'north', 'Orientation', 'horizontal', ...
        'FontSize', FS.legend + 5);
    apply_ieee_style(ax, FS);

    savefig_both(fig, basePath);
end

% ---- fig2: episode reward vs training episode (single 'orig' run per
% algo) -- a pale, unsmoothed trace behind a bold smoothed line, legend
% inside the plot at bottom-right. Mirrors the reference paper's training-
% curve figure; fig3 remains the fuller multi-seed/marker/band version.
% Also stamps each algo's mean/std/95% CI (from `data`, the SAME 20-episode
% evaluation results the fig2_reward_bar bars are built from) as a small
% colored text chip in the top-left, so this version carries the same
% summary numbers the bar version conveys visually via bar height + error
% bar, not just the training-progress curve. --------------------------------
function plot_fig2_reward_curve(outputDir, COL, FS, algos, data)
    trainFiles = struct( ...
        'SAC', 'results/sac_v17_training_metrics.json', ...
        'TD3', 'results/td3_v17_training_metrics.json', ...
        'PPO', 'results/ppo_v17_training_metrics.json');

    smoothWin = 100;

    % wider than the other single-panel figures -- leaves room for the
    % stats box that gets placed beside the legend, below.
    fig = figure('Visible', 'off', 'Position', [100 100 1750 650]);
    ax = axes('Parent', fig); hold(ax, 'on');

    algoHandles = gobjects(1, numel(algos));
    for a = 1:numel(algos)
        algo  = algos{a};
        col   = COL.(algo);
        fpath = trainFiles.(algo);
        if ~isfile(fpath)
            warning('Missing %s -- skipping %s', fpath, algo);
            continue;
        end
        d = jsondecode(fileread(fpath));
        y = d.episode_rewards;
        x = (1:numel(y))';

        % raw, unsmoothed trace in a pale version of the algo's color
        praw = plot(ax, x, y, '-', 'LineWidth', 0.8);
        praw.Color = [col, 0.25];

        if numel(y) >= smoothWin
            ySmooth = movmean(y, [smoothWin - 1, 0], 'Endpoints', 'discard');
            xSmooth = (smoothWin:numel(y))';
        else
            ySmooth = y;
            xSmooth = x;
        end
        algoHandles(a) = plot(ax, xSmooth, ySmooth, '-', 'Color', col, 'LineWidth', 2.4);
    end

    xlim(ax, [0 10000]);
    lgd = legend(ax, algoHandles, algos, 'FontSize', FS.legend + 11, 'Location', 'southeast');
    if FS.showTitles
        title(ax, 'Episode Reward vs Training Episode (DRL only)', ...
            'FontSize', FS.title + 6, 'FontWeight', 'bold');
    end
    xlabel(ax, 'Episode', 'FontSize', FS.label + 6, 'FontWeight', 'bold');
    ylabel(ax, 'Episode Reward', 'FontSize', FS.label + 6, 'FontWeight', 'bold');
    apply_ieee_style(ax, FS);
    set(ax, 'FontSize', FS.tick + 6);

    % ---- mean / std / 95% CI, as its own box placed directly to the LEFT
    % of the legend (same row, same height) -- NOT folded into the legend
    % labels (that made each entry too long and got clipped/garbled) and
    % NOT floated somewhere else on the figure (nothing guarantees that
    % spot stays clear of the curves). Finalize the layout first so the
    % legend's real position can be measured, then anchor off of it. -------
    drawnow;
    tighten_axes(fig);
    lgdPos = lgd.Position;   % figure-normalized [x y w h], now final

    % Wide enough (given the figure was widened to match) that the text
    % below fits on ONE line at this font size -- a narrower box let the
    % text wrap to 2 lines, which then overlapped the row above/below it
    % (and the legend) since rowH was sized for single-line text. -----------
    statsW = 0.44;
    gap    = 0.015;
    statsX = max(0.01, lgdPos(1) - statsW - gap);
    rowH   = lgdPos(4) / numel(algos);
    for a = 1:numel(algos)
        algo = algos{a};
        vals = data.(algo).episode_rewards;
        mu   = mean(vals);
        sd   = std(vals);
        nEp  = numel(vals);
        ci95 = 1.96 * sd / sqrt(nEp);
        txt = sprintf('%s \\mu=%.2f \\sigma=%.2f CI=[%.2f,%.2f]', ...
            algo, mu, sd, mu - ci95, mu + ci95);
        rowY = lgdPos(2) + lgdPos(4) - a * rowH;   % top row first, matching legend order
        ann = annotation(fig, 'textbox', [statsX, rowY, statsW, rowH], ...
            'String', txt, 'Color', COL.(algo), 'FontSize', FS.legend + 10, ...
            'FontWeight', 'bold', 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', ...
            'EdgeColor', 'none', 'FitBoxToText', 'off');
        ann.BackgroundColor = [1 1 1 0.85];
    end

    savefig_both(fig, fullfile(outputDir, 'fig2_reward'), true);
end

% ---- fig3: multi-seed training reward curves, IEEE style + shaded band -----
function plot_fig3_seeds(outputDir, COL, FS)
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
    smoothWin   = 100;  % also used as the rolling-std window for the shaded
                         % band, so upper/lower/xOrig always line up 1:1

    fig = figure('Visible', 'off', 'Position', [100 100 1400 750]);
    ax = axes('Parent', fig); hold(ax, 'on');

    algoHandles = gobjects(1, numel(algos));
    for a = 1:numel(algos)
        algo = algos{a};
        col  = COL.(algo);

        % ---- shaded uncertainty band around the 'orig' curve, drawn first
        % (so the lines/markers layer on top of it) ---------------------
        fpathOrig = trainFiles.(algo).orig;
        if isfile(fpathOrig)
            dOrig = jsondecode(fileread(fpathOrig));
            yOrig = dOrig.episode_rewards;
            if numel(yOrig) >= smoothWin
                ySmoothOrig = movmean(yOrig, [smoothWin - 1, 0], 'Endpoints', 'discard');
                xOrig = (smoothWin:numel(yOrig))';
                rollStd = movstd(yOrig, [smoothWin - 1, 0], 'Endpoints', 'discard');
                upper = ySmoothOrig + rollStd;
                lower = ySmoothOrig - rollStd;
                fb = fill(ax, [xOrig; flipud(xOrig)], [upper; flipud(lower)], col, ...
                    'EdgeColor', 'none');
                fb.FaceAlpha = 0.15;
            end
        end

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
            mStep = max(1, round(numel(x) / nMarkers));
            mStart = 1 + mod((s-1) * round(mStep/3), mStep);
            p = plot(ax, x, ySmooth, seedStyles{s}, 'LineWidth', 2.0, ...
                'Marker', seedMarkers{s}, 'MarkerSize', 7, ...
                'MarkerIndices', mStart:mStep:numel(x));
            p.Color = [col, seedAlphas(s)];
            p.MarkerEdgeColor = col;
        end
        algoHandles(a) = plot(ax, nan, nan, '-', 'Color', col, 'LineWidth', 2.6);
    end

    seedHandles = gobjects(1, numel(seedKeys));
    for s = 1:numel(seedKeys)
        seedHandles(s) = plot(ax, nan, nan, seedStyles{s}, 'Color', [0.4 0.4 0.4], ...
            'LineWidth', 2.0, 'Marker', seedMarkers{s}, 'MarkerSize', 7, ...
            'MarkerEdgeColor', [0.4 0.4 0.4]);
    end

    xlim(ax, [0 10000]);
    legend(ax, [algoHandles, seedHandles], [algos, seedLabels], ...
        'NumColumns', 6, 'FontSize', FS.legend + 11, 'Location', 'south', ...
        'Orientation', 'horizontal');
    if FS.showTitles
        title(ax, 'Training Reward (100-ep moving avg, 3 seeds each; shaded = rolling std)', ...
            'FontSize', FS.title + 4, 'FontWeight', 'bold');
    end
    xlabel(ax, 'Episode', 'FontSize', FS.label + 3, 'FontWeight', 'bold');
    ylabel(ax, 'Reward', 'FontSize', FS.label + 3, 'FontWeight', 'bold');
    apply_ieee_style(ax, FS);
    set(ax, 'FontSize', FS.tick + 3);

    savefig_both(fig, fullfile(outputDir, 'fig3_seeds'));
end

% ---- combined fig4-fig9: one 2x3 tiled figure, (a)-(f) sub-captions placed
% BELOW each tile (via xlabel, not a title above it), tight inter-tile
% spacing, hatched bars, percent-formatted ratio panels. --------------------
function plot_combined_eval(data, algos, COL, decompCol, HATCH, FS, basePath)
    % {metric_key, panel_title, ylabel, mode}; mode: '' | 'log' | 'percent'
    % '__decomp__' is handled specially below
    panelCfg = {
        'episode_avg_delay',              'End-to-End Delay',        'Avg Delay (ms)', '';
        '__decomp__',                      'Delay Decomposition',     'Delay (ms)',     '';
        'episode_cpu_viol_rate',          'CPU Violation Rate',      'Rate',           'percent';
        'episode_channel_overflow_ratio', 'Channel Overflow Ratio',  'Ratio',          'percent';
        'episode_timeout_ratio',          'Timeout Ratio',           'Ratio',          'percent';
        'episode_runtime_sec',            'Per-Episode Runtime',     'Seconds (log)',  'log';
    };
    letters = {'a', 'b', 'c', 'd', 'e', 'f'};
    n = numel(algos);
    barWidth = 0.65;

    fig = figure('Visible', 'off', 'Position', [100 100 1650 1080]);
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for idx = 1:size(panelCfg, 1)
        ax = nexttile(tl); hold(ax, 'on');
        key  = panelCfg{idx, 1};
        ttl  = panelCfg{idx, 2};
        ylab = panelCfg{idx, 3};
        mode = panelCfg{idx, 4};

        if strcmp(key, '__decomp__')
            keys = {'episode_avg_t_ul', 'episode_avg_t_comp', 'episode_avg_t_link'};
            M = zeros(n, 3);
            for i = 1:n
                for k = 1:3
                    M(i, k) = mean(data.(algos{i}).(keys{k}));
                end
            end
            refH = max(M(:));   % largest single segment in this panel -- shared hatch reference
            legHandles = gobjects(1, 3);
            totals = sum(M, 2);
            for i = 1:n
                hh = HATCH.(algos{i});
                yBottom = 0;
                for k = 1:3
                    seg = M(i, k);
                    p = patch(ax, ...
                        [i-barWidth/2, i+barWidth/2, i+barWidth/2, i-barWidth/2], ...
                        [yBottom, yBottom, yBottom+seg, yBottom+seg], ...
                        decompCol(k, :), 'FaceAlpha', 0.85, 'EdgeColor', decompCol(k, :)*0.55, ...
                        'LineWidth', 1.1);
                    if i == 1
                        legHandles(k) = p;
                    end
                    if seg > 1e-9
                        draw_hatch_bar(ax, i-barWidth/2, yBottom, barWidth, seg, ...
                            hh.style, [0 0 0 0.45], 1.0, hh.density, false, refH);
                    end
                    yBottom = yBottom + seg;
                end
                text(ax, i, totals(i), sprintf('%.3f', totals(i)), 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'bottom', 'FontSize', FS.value - 2, 'FontWeight', 'bold');
            end
            legend(ax, legHandles, {'t_{ul}', 't_{comp}', 't_{link}'}, ...
                'FontSize', FS.legend - 2, 'Location', 'northoutside', 'Orientation', 'horizontal');
        else
            isLog = strcmp(mode, 'log');
            isPercent = strcmp(mode, 'percent');
            means = zeros(1, n); stds = zeros(1, n);
            for i = 1:n
                vals = data.(algos{i}).(key);
                if isPercent
                    vals = vals * 100;   % 0-1 fraction -> plain 0-100 number, no '%' suffix
                end
                means(i) = mean(vals);
                stds(i)  = std(vals);
            end
            if isLog
                yBase = min(means) / 3;
            else
                yBase = 0;
            end
            refH = max(abs(means - yBase));   % tallest bar's height -- shared hatch reference
            for i = 1:n
                hh = HATCH.(algos{i});
                draw_one_bar(ax, i, yBase, means(i), barWidth, COL.(algos{i}), hh.style, hh.density, isLog, refH);
            end
            errorbar(ax, 1:n, means, stds, 'k', 'LineStyle', 'none', ...
                'LineWidth', 1.3, 'CapSize', 6);
            for i = 1:n
                text(ax, i, means(i) + stds(i), sprintf('%.3f', means(i)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', FS.value - 2, 'FontWeight', 'bold');
            end
            if isLog
                set(ax, 'YScale', 'log');
                ylim(ax, [yBase, max(means + stds) * 2]);
            end
        end

        if ~strcmp(mode, 'log')
            % Every metric in this combined figure (delay, decomp, ratios)
            % is non-negative -- pin the bottom to 0 so MATLAB's auto-tick
            % padding doesn't sneak a negative range in below it.
            yl = ylim(ax);
            ylim(ax, [0, yl(2)]);
        end

        set(ax, 'XTick', 1:n, 'XTickLabel', algos);
        xlim(ax, [0.5, n + 0.5]);
        ylabel(ax, ylab, 'FontSize', FS.label - 2, 'FontWeight', 'bold');
        apply_ieee_style(ax, FS);
        set(ax, 'FontSize', FS.tick - 2);
        % sub-caption BELOW the panel (xlabel sits under the tick labels,
        % matching the reference's "(a) ..." placement under each tile)
        xlabel(ax, sprintf('(%s) %s', letters{idx}, ttl), ...
            'FontSize', FS.label, 'FontWeight', 'bold');
    end

    if FS.showTitles
        title(tl, 'SAC vs TD3 vs PPO vs Greedy vs GA -- Evaluation Comparison', ...
            'FontSize', FS.title, 'FontWeight', 'bold');
    end

    savefig_both(fig, basePath);
end
