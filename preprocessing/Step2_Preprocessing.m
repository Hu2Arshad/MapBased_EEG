%% ====================================================================
% Step 2: Preprocessing  Resample, Bandpass, Notch, ICA
% ====================================================================
clear; clc;

target_sfreq = 200;
locutoff     = 0.53;     
hicutoff     = 30;       
notch_freq   = 60;      
iclabel_thr  = 0.9;

ICA_SEED = str2double(getenv('ICA_SEED'));                           % ICA RNG seed (reproducible runica); default 0
if isnan(ICA_SEED); ICA_SEED = 0; end
fprintf('ICA_SEED = %d\n', ICA_SEED);

EEGLAB_DIR = 'D:\2026\Prof hsu project\Dataset\Data1\eeglab2025.1.0';
REPO_ROOT  = fileparts(fileparts(mfilename('fullpath')));            % <repo> (parent of preprocessing/)
RAW_DIR    = fullfile(REPO_ROOT, 'data', 'converted');              % input: Step 1 output
OUTPUT_DIR = fullfile(REPO_ROOT, 'data', 'preprocessed');           % output: Step 3 input
groups     = {'HC','MCI','SCD'};                                                   

addpath(EEGLAB_DIR);
eeglab nogui;
eeglab_dir    = fileparts(which('eeglab.m'));
chan_loc_file = fullfile(eeglab_dir, 'plugins', 'dipfit', 'standard_BEM', 'elec', 'standard_1005.elc');

%% Discover files
file_list = [];
for g = 1:length(groups)
    group_in  = fullfile(RAW_DIR, groups{g});
    group_out = fullfile(OUTPUT_DIR, groups{g});
    if ~exist(group_out, 'dir'); mkdir(group_out); end
    if ~exist(group_in, 'dir'); warning('missing %s', group_in); continue; end
    files = dir(fullfile(group_in, 'sub*.set'));
    for f = 1:length(files)
        file_list(end+1).name    = files(f).name; %#ok<SAGROW>
        file_list(end).folder    = files(f).folder;
        file_list(end).group     = groups{g};
        file_list(end).out_path  = group_out;
    end
end
fprintf('Found %d file(s).\n', length(file_list));

%% Parallel pool (serial fallback)
try
    if license('test','Distrib_Computing_Toolbox') && isempty(gcp('nocreate'))
        parpool;
    end
catch poolME
    warning('parpool failed (%s) -- running serially', poolME.message);
end

%% Main loop
parfor i = 1:length(file_list)
    currFile = file_list(i).name;
    out_dir  = file_list(i).out_path;
    [~, fileName, ~] = fileparts(currFile);
    outName  = sprintf('Pre_processed_%s.set', fileName);
    if exist(fullfile(out_dir, outName), 'file')
        fprintf('  SKIP %s\n', currFile); continue;
    end
    try
        % 1. Load + channel locations
        EEG = pop_loadset('filename', currFile, 'filepath', file_list(i).folder);
        EEG = pop_chanedit(EEG, 'lookup', chan_loc_file);

        % 2. Notch 60 Hz + bandpass 0.53-30 Hz
        EEG = pop_eegfiltnew(EEG, 'locutoff', notch_freq-1, 'hicutoff', notch_freq+1, 'revfilt', 1);
        EEG = pop_eegfiltnew(EEG, 'locutoff', locutoff, 'hicutoff', hicutoff);

        % 3. Resample to 200 Hz
        if EEG.srate ~= target_sfreq
            EEG = pop_resample(EEG, target_sfreq);
        end

        % 4. Average reference 
        EEG = pop_reref(EEG, []);

        % 5. ICA -- default rank  (seeded per file so runica is reproducible; parfor workers need rng set inside the loop)
        rng(ICA_SEED + i, 'twister');
        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
        % 6. ICLabel -> remove muscle/eye/heart >= thr
        EEG = pop_iclabel(EEG, 'default');
        EEG = pop_icflag(EEG, [NaN NaN; ...          % Brain
                               iclabel_thr 1; ...    % Muscle
                               iclabel_thr 1; ...    % Eye
                               iclabel_thr 1; ...    % Heart
                               NaN NaN; ...          % Line Noise
                               NaN NaN; ...          % Channel Noise
                               NaN NaN]);            % Other
        % --- ICA DIAGNOSTIC ---
        cls = EEG.etc.ic_classification.ICLabel.classifications;   % ncomp x 7 [Brain Muscle Eye Heart Line Chan Other]
        [~, am] = max(cls, [], 2);
        art = max(cls(:,2:4), [], 2);                              % max muscle/eye/heart prob per comp
        var_before = sum(var(EEG.data,0,2));
        rej = find(EEG.reject.gcompreject);
        if ~isempty(rej); EEG = pop_subcomp(EEG, rej, 0); end
        var_after = sum(var(EEG.data,0,2));
        fprintf('ICADIAG %s ncomp=%d nbrain=%d nmuscle=%d neye=%d nheart=%d nart50=%d nart70=%d nart90=%d nremoved=%d pctvar=%.3f\n', ...
            currFile, size(cls,1), sum(am==1), sum(am==2), sum(am==3), sum(am==4), ...
            sum(art>0.5), sum(art>0.7), sum(art>0.9), numel(rej), 100*(var_before-var_after)/max(var_before,eps));
        fprintf('OK %s | compsRej=%d pnts=%d\n', currFile, numel(rej), EEG.pnts);

        % 8. Save
        EEG = pop_saveset(EEG, 'filename', outName, 'filepath', out_dir, 'check','on', 'savemode','onefile');
    catch ME
        warning('FAILED %s: %s', currFile, ME.message);
    end
end

fprintf('\n==== Step 2 COMPLETE ====\n');
