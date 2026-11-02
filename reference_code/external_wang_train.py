"""Fixed, test-independent training of the unchanged Wang policy/environment.

Default algorithm hyperparameters match upstream. CPU + controlled native pool
generation + deterministic environment RNG are portability/reproducibility fixes.
Run in its own process; all training instances are official synthetic generator R.
"""
import argparse
import json
import os
import time
import hashlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from external_wang import native, SeededEnvironment, VRPTWGeneratorR, BASE, COMMIT
from ppo.instance_pool import InstancePool
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from external_affinity import bind_all_threads


class SyntheticPool(InstancePool):
    """Native generated data, stored to disk; sequential generation avoids the
    upstream pool's concurrent np.random.seed global-RNG race."""
    def __init__(self, root, dimension, seed):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        generator = VRPTWGeneratorR(dimension, 64, 16, dimension, 5, 15, 50)
        super().__init__(generator, pool_size=128, refresh_threshold=4096,
                         sampling_memory=128, max_workers=1, chunk_size=32, seed=seed)

    def initialize(self):
        first_seed = self.current_seed
        super().initialize()
        for i, data in enumerate(self.instances):
            np.savez_compressed(self.root / f'seed_{first_seed+i}.npz', **data)


class Progress(BaseCallback):
    def __init__(self, output, started, save_freq):
        super().__init__()
        self.output = Path(output)
        self.started = started
        self.save_freq = save_freq
        self.last_save = 0

    def _on_step(self):
        if self.n_calls % 100 == 0:
            progress = {'timesteps': self.num_timesteps, 'elapsed_s': time.perf_counter()-self.started}
            (self.output / 'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
            print(json.dumps(progress), flush=True)
        if self.num_timesteps - self.last_save >= self.save_freq:
            self.model.save(str(self.output / f'checkpoint_{self.num_timesteps}'))
            self.last_save = self.num_timesteps
        return True


def factory(root, dimension, seed, rank):
    def create():
        cpu_binding=bind_all_threads()
        torch.set_num_threads(1)
        cpu_binding=bind_all_threads()
        native.ppo_config.dimension = dimension
        # Independent per-worker RNG/pool, all seeds disjoint from test seeds.
        pool = SyntheticPool(Path(root) / f'worker_{rank:02d}', dimension, seed + rank * 100000)
        if cpu_binding:
            (Path(root)/f'worker_{rank:02d}'/'cpu_binding.json').write_text(json.dumps(cpu_binding,indent=2),encoding='utf8')
        return SeededEnvironment(pool, seed + rank * 100000)
    return create


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', default=str(BASE/'models/Wang_official_n100'))
    p.add_argument('--timesteps',type=int,default=1600000)
    p.add_argument('--envs',type=int,default=32)
    p.add_argument('--n-steps',type=int,default=1000)
    p.add_argument('--dimension',type=int,default=101)
    p.add_argument('--seed',type=int,default=3333)
    p.add_argument('--train-threads',type=int,default=4)
    p.add_argument('--cpu-group',type=int,default=None)
    p.add_argument('--cpu-mask',type=lambda value:int(value,0),default=None)
    args=p.parse_args()
    if args.cpu_group is not None:
        os.environ['REVISION_WANG_GROUP']=str(args.cpu_group)
    if args.cpu_mask is not None:
        os.environ['REVISION_WANG_MASK']=str(args.cpu_mask)
    cpu_binding=bind_all_threads()
    torch.set_num_threads(args.train_threads)
    if cpu_binding:
        # Create Torch's lazy thread pool before verifying all existing threads.
        torch.ones((256,256)).matmul(torch.ones((256,256)))
        cpu_binding=bind_all_threads()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if cpu_binding:
        (output/'cpu_binding.json').write_text(json.dumps(cpu_binding,indent=2),encoding='utf8')
    t0=time.perf_counter()
    config=vars(args)|{'upstream_commit':COMMIT,'batch_size':128,'n_epochs':10,'learning_rate':0.001,
        'ent_coef':0.001,'architecture':[512,256,128],'device':'cpu','generator':'native R',
        'test_instances_seen':0,'selection_rule':'final checkpoint; no test selection',
        'pool_portability_changes':['max_workers=1 avoids global np.random.seed races',
            'independent seeded 128-instance native pool per worker'],
        'note':'Native instance pool weighted sampling/memory/refresh and native generator retained.'}
    (output/'training_config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    factories=[factory(output/'training_data',args.dimension,args.seed,i) for i in range(args.envs)]
    env=SubprocVecEnv(factories) if args.envs>1 else DummyVecEnv(factories)
    model=PPO('MultiInputPolicy',env,n_steps=args.n_steps,batch_size=128,n_epochs=10,
        learning_rate=lambda p: p*0.001,ent_coef=0.001,verbose=1,device='cpu',seed=args.seed,
        policy_kwargs={'net_arch':dict(pi=[512,256,128],vf=[512,256,128]),'activation_fn':torch.nn.ReLU})
    model.learn(total_timesteps=args.timesteps,callback=Progress(output,t0,160000))
    model.save(str(output/'final_model'))
    env.close()
    checkpoint = output/'final_model.zip'
    data_files = sorted((output/'training_data').rglob('*.npz'))
    data_manifest = [{'path':str(f.relative_to(output)),
                      'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in data_files]
    manifest_path = output/'training_data_manifest.json'
    manifest_path.write_text(json.dumps(data_manifest,indent=2),encoding='utf8')
    complete={'status':'completed','timesteps':model.num_timesteps,'elapsed_s':time.perf_counter()-t0,
              'checkpoint_path':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'training_config_path':str(output/'training_config.json'),
              'training_data_manifest_path':str(manifest_path),'training_instance_count':len(data_files),
              'generator':'native Wang R','seed_formula':'3333 + worker_rank*100000 + pool_generation*128 + instance_index',
              'test_instances_seen':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    tmp=output/'completed.json.tmp'
    tmp.write_text(json.dumps(complete,indent=2),encoding='utf8')
    tmp.replace(output/'completed.json')


if __name__=='__main__':
    main()
