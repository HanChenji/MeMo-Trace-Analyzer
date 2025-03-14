#!/usr/bin/env python
import argparse
import os, shutil
import numpy as np
import json as js
from scripts import utils
from scripts.utils import set_logger, NumpyEncoder


def parse_args():
    if 'ZSIMDIR' not in os.environ:
        os.environ['ZSIMDIR'] = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.environ['ZSIMDIR']

    parser = argparse.ArgumentParser(description="Run ZSim Arguments")
    parser.add_argument("--base-dir", type=str, default=base_dir, help="Base directory")
    parser.add_argument("--job-nums", "-n", dest="job_nums", type=int, default=1, help="number of jobs")
    parser.add_argument("--threads", "-j", dest="threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--job-list", "-l", dest="job_list", type=str, nargs='+', help="file of the job list")
    parser.add_argument("--program","-p",dest="program", type=str, help="name of workload")
    parser.add_argument("--task", "-t",dest="task", type=str, help="type of analysis routine")
    parser.add_argument("--config", type=str, default="simple", help="Path to the config file")
    # options for profiling
    parser.add_argument("--profiling-force", action="store_true", help="Force to do profiling")
    parser.add_argument("--profiling-order", type=int, default=0, help="Order of the profiling routine")
    parser.add_argument("--profiling-slice-size", type=int, default=100000000, help="Size of the slice during profiling")
    parser.add_argument("--profiling-emit-first", type=bool, default=True, help="Emit the first slice")
    parser.add_argument("--profiling-emit-last", type=bool, default=True, help="Emit the last slice")
    # options for clustering
    parser.add_argument("--clustering-j", type=int, default=1, help="Seed for random projection")
    parser.add_argument("--clustering-signature", type=str, default="bbv", help="Signature type")
    parser.add_argument("--clustering-algo", type=str, default="kmeans", help="Clustering algorithm")
    parser.add_argument("--clustering-slice-size", type=int, default=100000000, help="Size of the slice during clustering")
    parser.add_argument("--clustering-nclusters-min", type=int, default=1, help="Minimum Number of clusters")
    parser.add_argument("--clustering-nclusters-max", type=int, default=30, help="Maximum Number of clusters")
    parser.add_argument("--clustering-distance", type=str, default='euclidean', help="Distance metric")
    parser.add_argument("--clustering-proj-dim", type=int, default=15, help="Dimension of the projection")
    parser.add_argument("--clustering-runs", type=int, default=100, help="Number of runs for clustering")
    parser.add_argument("--clustering-iters", type=int, default=20, help="Number of iterations for clustering")
    parser.add_argument("--clustering-kmeans-init", type=str, default="kmeans++", help="Initialization method")
    parser.add_argument("--clustering-pca-var", type=float, default=0.99, help="Variance to keep for PCA")
    parser.add_argument("--clustering-init-seed", type=int, default=493575226, help="Seed for random number generator")
    parser.add_argument("--clustering-proj-seed", type=int, default=2042712918, help="Seed for random projection")
    parser.add_argument("--clustering-nan-value", type=float, default=0, help="Value to replace NaN")
    parser.add_argument("--clustering-force", action="store_true", help="Force to do clustering")
    # options for analysis
    parser.add_argument("--analysis-bic", type=float, default=0.95, help="BIC threshold")
    parser.add_argument("--analysis-maxK", type=int, default=30, help="maxK")

    config = vars(parser.parse_args())

    config['zsim_bin'] = os.path.join(config['base_dir'], 'build/opt/zsim')


    return config

class ZSimRunner:
    def __init__(self, config:dict):
        self.config = config

    def do_profiling(self):
        import libconf
        import tempfile

        # check if profiling is already done
        zsim_log = os.path.join(self.config['profiling_dir'], 'zsim.log.0')
        # check if zsim_log contains 'Finished, code 0'
        if os.path.exists(zsim_log) and 'Finished, code 0' in open(zsim_log).read() and not self.config['profiling_force']:
            self.logger.info(f"Profiling already done for {self.config['program']}")
            return

        with open(os.path.join('config', f"{self.config['config']}.cfg"), 'r') as f:
            zsim_cfg = libconf.load(f)
        zsim_cfg['sim'] = {
            'taylorsim' : self.config['profiling_options'],
            'outputDir' : self.config['profiling_dir'],
            'logToFile' : True,
            'strictConfig' : False,
            'parallelism' : 1,
            'schedQuantum' : 100000000,
        }
        zsim_cfg['process0'] = {
            'command' : utils.get_app_option(self.config, 'command'),
        }
        if utils.check_app_option(self.config, 'stdin'):
            zsim_cfg['process0']['input'] = utils.get_app_option(self.config, 'stdin')
        if utils.check_app_option(self.config, 'loader'):
            zsim_cfg['process0']['loader'] = utils.get_app_option(self.config, 'load')
        if utils.check_app_option(self.config, 'env'):
            zsim_cfg['process0']['env'] = utils.get_app_option(self.config, 'env')
        if utils.check_app_option(self.config, 'heap'):
            zsim_cfg['sim']['gmMBytes'] = int(utils.get_app_option(self.config, 'heap'))

        run_dir = tempfile.mkdtemp()
        with open(os.path.join(run_dir, 'zsim.cfg'), 'w') as f:
            libconf.dump(zsim_cfg, f)
        profling_cmd = f"{self.config['zsim_bin']} zsim.cfg"

        # link files and dirs from self.config['app_dir] to run_dir
        utils.linkfiles(self.config['app_dir'], run_dir)
        # link the binary to run_dir and name it as base.exe
        os.symlink(utils.get_app_bin(self.config), os.path.join(run_dir, 'base.exe'))

        # run!
        self.logger.info(f"Running {profling_cmd}")
        utils.ex_log(profling_cmd, log_file=self.config['log_file'], cwd=run_dir)

        # remove the run_dir
        shutil.rmtree(run_dir)

        # record the self.config json
        import json as js
        with open(os.path.join(self.config['profiling_dir'], 'profiling.config'), 'w') as f:
            js.dump(self.config, f, indent=4, cls=utils.NumpyEncoder)

    def do_clustering(self):
        options = self.config['clustering_options']

        self.logger.info("Loading Signature...")
        if options['signature'] == 'bbv':
            X = utils.load_bbv(self.config)
        elif options['signature'] == 'MeMo':
            X = utils.load_MeMo(self.config)
        else:
            raise ValueError(f"Unknown signature type {options['signature']}")

        nclusters_candis = list(range(options['nclusters_min'], options['nclusters_max']+1))

        def cluster_task(ncluster):
            self.do_clustering_per_ncluster(X, ncluster)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['clustering_j']) as executor:
            executor.map(cluster_task, nclusters_candis)

    def do_clustering_per_ncluster(self, X, ncluster):
        import json as js
        from scripts.kmeans import KMeans

        options = self.config['clustering_options']

        roi_file = os.path.join(
            self.config['clustering_dir'],
            "rois_"+utils.clustering_suffix(options, numK=ncluster)
        )

        if os.path.exists(roi_file) and os.path.getsize(roi_file) and not self.config['clustering_force']:
            self.logger.info(f"Clustering has already done for {roi_file}")
            return

        attempts = {}
        scores   = []

        self.logger.info(f"Clustering for {ncluster} clusters...")
        for idx in range(options['runs']):
            self.logger.info(f"Clustering for {idx} run...")
            kmeans = KMeans(ncluster, options, idx, self.logger)
            kmeans.fit(X)
            attempts[f'run_{idx}'] = kmeans.save()
            scores.append(kmeans.score)
        bset_clustering = attempts[f'run_{np.argmax(scores)}']

        # dump attempts
        with open(roi_file, 'w') as f:
            js.dump(bset_clustering, f, indent=4, cls=utils.NumpyEncoder)

        return

    def do_analysis(self):
        from scripts.H5Reader import H5Reader

        def get_rois(ncluster):
            rois_file = os.path.join(
                self.config['clustering_dir'], 
                "rois_" + utils.clustering_suffix(self.config['clustering_options'], numK=ncluster)
            )
            with open(rois_file) as f:
                return js.load(f)

        clusers_range = range(1, self.config['analysis_maxK']+1)
        rois = {ncluster: get_rois(ncluster) for ncluster in clusers_range}
        scores = {ncluster: rois[ncluster]['score'] for ncluster in clusers_range}

        min_score = min(scores.values())
        max_score = max(scores.values())
        threshold = min_score + (max_score - min_score) * self.config['analysis_bic']
        filtered = {k: v for k, v in scores.items() if v >= threshold}

        K =  min(filtered, key=filtered.get)
        K_rois = rois[K]['centroids']
        K_weights = rois[K]['weights']

        core_cpis = H5Reader(self.config).cpis()
        true_avg_cpi        = np.mean(core_cpis)
        estimated_avg_cpi   = np.sum(core_cpis[K_rois] * K_weights) / np.sum(K_weights)
        estimated_cpi_error = np.abs(estimated_avg_cpi - true_avg_cpi) / true_avg_cpi

        self.logger.info(f"Signature: {self.config['clustering_signature']}")
        self.logger.info(f"\tSelected program interval number: {K}")
        self.logger.info(f"\tEstimated CPI error: {estimated_cpi_error*100:.2f}%") 

    def run(self, program):
        self.config['program'] = program
        self.config = utils.update_config(self.config)

        suffix = "with_"+utils.clustering_suffix(self.config['clustering_options'], suffix='log')
        self.config['log_file'] = os.path.join(self.config['logs_dir'],
            '.'.join([self.config['task'], self.config['program'], self.config['config'], suffix])
        )
        self.logger = set_logger(self.config['log_file'])

        if self.config['task'] == 'profiling':
            self.do_profiling()
        elif self.config['task'] == 'clustering':
            self.do_clustering()
        elif self.config['task'] == 'analysis':
            self.do_analysis()
        else:
            raise ValueError("Invalid analysis routine")

def single_run(program, config):
    zsim_runner = ZSimRunner(config)
    zsim_runner.run(program)

if __name__ == "__main__":
    config = parse_args()

    programs = []
    if config['job_list']:
        for job_file in config['job_list']:
            with open(job_file, 'r') as f:
                for line in f:
                    if not line.startswith('#') and line.strip() != '':
                        programs.append(line.strip())
    else:
        programs.append(config['program'])

    # parallelize programs
    import multiprocessing
    pool = multiprocessing.Pool(config['job_nums'])
    pool.starmap(single_run, [(program, config) for program in programs])
    pool.close()
    pool.join()
