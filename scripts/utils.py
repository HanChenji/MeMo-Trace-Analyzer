import os
import errno
import subprocess
import glob
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def std_and_pca(data, variance=0.99):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # convert each column with mean value 0 and variance 1
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    pca = PCA(variance)
    data = pca.fit_transform(data)

    return data

def post_js(data):
    data_shifted = data - data.min(axis=0)
    data_normalized = data_shifted / data_shifted.sum(axis=1, keepdims=True)

    return data_normalized

def load_bbv(config):
    from scripts.loader import Loader
    from scripts.Normalizer import Normalizer
    from scripts.RandomLinearProjection import RandomLinearProjection
    from sklearn import pipeline

    tb_file = os.path.join(config['bbv_dir'], 'taylorblock.T.0.order0.bb.gz')
    loader = Loader(tb_file, n_threads=config['threads'])
    bbv = loader.load()

    # save slices
    slices_file = os.path.join(config['bbv_dir'], 'slices_num.npy')
    if not os.path.exists(slices_file):
        with open(slices_file, 'wb') as f:
            np.save(f, bbv.shape[0])

    options = config['clustering_options']
    handles = [
        ('normalizer', Normalizer()),
        ('projection', RandomLinearProjection(options['proj_dim'], random_state=options['proj_seed'])),
    ]
    bbv = pipeline.Pipeline(handles).fit_transform(bbv)
    
    if options['distance'] == 'jensenshannon':
        bbv = post_js(bbv)

    return bbv

def load_MeMo(config):
    from scripts.H5Reader import H5Reader

    def get_stat(model):
        reader_config = {
            'config': 'MeMo',
            'bbv_dir': config['bbv_dir'],
            'profiling_dir': config['profiling_dir'].replace(config['config'], model),
        }
        core_stats = H5Reader(reader_config)
        return core_stats

    arch_range = ['x1', 'x2', 'x4', 'x8', 'x16']

    l1d_miss_rate = []
    l2_miss_rate  = []
    llc_miss_rate = []
    mem_avg_lat   = []
    for model in [f'CacheModel{arch}' for arch in arch_range]:
        core_stats = get_stat(model)
        l1d_miss_rate.append(core_stats.l1d_miss_rate())
        l2_miss_rate.append (core_stats.l2_miss_rate())
        llc_miss_rate.append(core_stats.llc_miss_rate())
        mem_avg_lat.append(core_stats.get_cache_subsystem_avg_lat())
    l1d_miss_rate = np.array(l1d_miss_rate).T
    l2_miss_rate  = np.array(l2_miss_rate).T
    llc_miss_rate = np.array(llc_miss_rate).T
    mem_avg_lat   = np.array(mem_avg_lat).T

    fetch_stalls = []
    for model in ['FetchModel-widthx1', 'FetchModel-widthx2', 'FetchModel-x4', 'FetchModel-widthx8', 'FetchModel-widthx16']:
        core_stats = get_stat(model)
        fetch_stalls.append(core_stats.fetch_stalls())
    fetch_stalls = np.array(fetch_stalls).T

    l1i_miss_rate = []
    for model in ['FetchModel-cachex1', 'FetchModel-cachex2', 'FetchModel-x4', 'FetchModel-cachex8', 'FetchModel-cachex16']:
        core_stats = get_stat(model)
        l1i_miss_rate.append(core_stats.l1i_miss_rate())
    l1i_miss_rate = np.array(l1i_miss_rate).T

    br_misses = []
    for model in ['FetchModel-bpx1', 'FetchModel-bpx2', 'FetchModel-x4', 'FetchModel-bpx8', 'FetchModel-bpx16']:
        core_stats = get_stat(model)
        br_misses.append(core_stats.br_misses())
    br_misses = np.array(br_misses).T

    issue_stalls = []
    for model in [f'IssueModel{arch}' for arch in arch_range]:
        core_stats = get_stat(model)
        issue_stalls.append(core_stats.issue_stalls())
    issue_stalls = np.array(issue_stalls).T

    icount = get_stat('IssueModelx1').core_instrs()
    spawned_icount = np.array([icount] * br_misses.shape[1]).T

    br_mpki      = br_misses    / spawned_icount * 1000
    fetch_stalls = fetch_stalls / spawned_icount
    issue_stalls = issue_stalls / spawned_icount

    ours = np.column_stack((
        l1d_miss_rate, l2_miss_rate, llc_miss_rate, mem_avg_lat,
        l1i_miss_rate, fetch_stalls, br_mpki,
        issue_stalls,
    ))

    ours = std_and_pca(ours, variance=config['clustering_pca_var'])

    if config['clustering_distance'] == 'jensenshannon':
        ours = post_js(ours)

    return ours

def set_logger(log_file):
    import logging
    # check if log_file exists, if so rename
    if os.path.exists(log_file):
        os.rename(log_file, log_file + '.old')
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger

def clustering_suffix(options, suffix='json', numK=None):
    name = '_'.join([
        options['algo'],
        options['distance'],
        f'N{options["slice_size"]}',
        options['signature'],
        f'K{numK}',
    ])

    if options['signature'] == 'bbv':
        name += f'_dim{options["proj_dim"]}'
    elif options['signature'] == 'MeMo':
        name += f'_pca{options["pca_var"]}'

    if options['init_seed'] != 493575226:
        name += f'_seed{options["init_seed"]}'

    name += f'_runs{options["runs"]}'
    name += f'_iters{options["iters"]}'
    name =  '.'.join([name, suffix])

    return name

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def check_dependency(config:dict):
    # check if workload name is valid
    if config['program'].count('-') != 2:
        raise ValueError("Invalid workload name")
    if config['clustering_distance'] not in ['jensenshannon', 'euclidean', 'cosine', 'cityblock']:
        raise ValueError("Invalid distance metric")
    if config['clustering_algo'] not in ['kmeans', 'spectralc', 'aggc','bisecting', 'gaussian', 'birch']:
        raise ValueError("Invalid clustering algorithm")

def update_config(config:dict):
    check_dependency(config)

    config['data_dir'] = os.path.join(config['base_dir'], 'data')
    mkdir_p(config['data_dir'])
    config['logs_dir'] = os.path.join(config['base_dir'], 'logs')
    mkdir_p(config['logs_dir'])
    config['apps_dir'] = os.path.join(config['base_dir'], 'apps')

    config['bm_suite'] = config['program'].split('-')[0]
    config['bm_name' ] = config['program'].split('-')[1]
    config['bm_input'] = config['program'].split('-')[2]
    config['bm_fullname'] = get_fullname(config['bm_name'])
    path_suffix = os.path.join(config['bm_suite'], config['bm_name'], 'ref', f"input_{config['bm_input']}")

    config['bbv_dir']  = os.path.join(config['data_dir'], 'profiling', 'BbvProfiler', path_suffix)

    config['profiling_dir']  = os.path.join(config['data_dir'], 'profiling', config['config'], path_suffix)
    mkdir_p(config['profiling_dir'])

    config['clustering_dir'] = os.path.join(config['data_dir'], 'clustering', path_suffix)
    mkdir_p(config['clustering_dir'])

    config['analysis_dir'] = os.path.join(config['data_dir'], 'analysis', path_suffix)
    mkdir_p(config['analysis_dir'])

    config['app_dir'] = os.path.join(config['apps_dir'], config['bm_suite'], config['bm_fullname'], 'ref')
    config['app_cfg'] = os.path.join(config['app_dir'], '.'.join([config['bm_fullname'], config['bm_input'], 'cfg']))

    config['profiling_options'] = {
        'slice_size' : config['profiling_slice_size'],
        'emit_first' : config['profiling_emit_first'],
        'emit_last'  : config['profiling_emit_last' ],
    }

    config['clustering_options'] = {
        'signature'  : config['clustering_signature'],
        'algo'       : config['clustering_algo'],
        'slice_size' : config['clustering_slice_size'],
        'nclusters_min': config['clustering_nclusters_min'],
        'nclusters_max': config['clustering_nclusters_max'],
        'distance'   : config['clustering_distance'],
        'proj_dim'   : config['clustering_proj_dim'],
        'pca_var'    : config['clustering_pca_var'],
        'runs'       : config['clustering_runs'],
        'iters'      : config['clustering_iters'],
        'threads'    : config['threads'],
        'nan_value'  : config['clustering_nan_value'],
        'kmeans_init': config['clustering_kmeans_init'],
        'init_seed'  : config['clustering_init_seed'],
        'proj_seed'  : config['clustering_proj_seed'],
    }

    return config

def get_app_option(config, option:str):
    with open(config['app_cfg']) as f:
        for line in f:
            if line.startswith(option):
                return line.split(':', 1)[1].strip()
    raise ValueError(f"No {option} in the app config")

def check_app_option(config, option:str):
    with open(config['app_cfg']) as f:
        for line in f:
            if line.startswith(option):
                return True
    return False

def list_executable_elf_files(directory):
    # List all files in the directory
    files_in_directory = os.listdir(directory)
    # Filter the list to only include executable ELF files
    executable_elf_files = [file for file in files_in_directory if is_executable_elf(os.path.join(directory, file))]
    return executable_elf_files

def is_executable_elf(file_path):
    # Use the 'file' command to get the file type
    file_type = subprocess.check_output(['file', '-b', file_path]).decode()
    # Check if the file type is 'ELF' and 'executable'
    return 'ELF' in file_type and 'executable' in file_type

def get_app_bin(config):
    bin_dir = os.path.join(config['apps_dir'], config['bm_suite'], config['bm_fullname'])
    candidate_bins = list_executable_elf_files(bin_dir)
    if len(candidate_bins) != 1:
        raise ValueError(f"Multiple or no executable files in {bin_dir}")
    return os.path.join(bin_dir, candidate_bins[0])

def linkfiles(src, dest, exclude=None):
    for i in glob.iglob(os.path.join(src, "*")):
        if exclude and exclude in os.path.basename(i):
            continue
        d = os.path.join(dest, os.path.basename(i))
        try:
            os.symlink(i, d)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("[PREPROCESS] File", i, "exists; skipping.")

def ex(cmd, cwd=".", env=None):
    proc = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=env)
    proc.communicate()

def ex_log(cmd, log_file, cwd=".", env=None):
    import time
    with open(log_file, "a") as f:
        f.write("[cwd] %s\n" % cwd)
        f.write("[command] %s\n" % cmd)
        f.write("[begin time] %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    cmd += f" 2>&1 | tee -a {log_file}"
    ex(cmd, cwd, env)
    with open(log_file, "a") as f:
        f.write("[end time] %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))

def get_fullname(raw):
    name_mapper = {
        "gcc_r": "502.gcc_r",
    }
    return name_mapper[raw]
