"""Fully recorded source training. Test instances are never used for selection."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse
from pathlib import Path
import time
from experiment_io import ROOT,atomic_json,read_json,sha256,source_manifest,timestamp,bind_slot

def train(job):
    slots=read_json(ROOT/'reports'/'machine.json')['physical_core_slots']
    bind_slot(slots[int(job.get('slot',len(slots)-1))])
    import torch
    from controlled_engine import create_agent,run_trial,save_checkpoint,warmup_backend
    from validate_routes import validate_routes
    mode,problem,method=job['mode'],job['problem'],job['method']
    jid=job['job_id']; dest=ROOT/'models'/jid
    dest.mkdir(parents=True,exist_ok=True)
    final=dest/'final.pth'
    done=dest/'completed.json'
    if done.exists() and final.exists():
        status=read_json(done)
        if status['checkpoint_sha256']!=sha256(final): raise RuntimeError('Final training checkpoint changed')
        if read_json(dest/'protocol.json')['job']!=job: raise RuntimeError('Completed training protocol differs from requested job')
        return status
    dataset=read_json(ROOT/'configs'/('pretrain_instances.json' if mode=='light' else 'source_training_instances.json'))
    dataset=[r for r in dataset if r['problem']==problem]
    source_files={r['id']:r['sha256'] for r in dataset}
    for r in dataset:
        if sha256(r['path'])!=r['sha256']:raise RuntimeError('Training data modified')
    # Sequential episodes in fixed source order. Both epochs are explicit.
    tasks=[(epoch,r) for epoch in range(2) for r in dataset]
    warmup_backend(problem)
    agent=create_agent(problem,method,seed=job['training_seed'])
    optim_names=['opt_enc','opt_critic']+(['opt_joint'] if method=='AC' else ['opt_ad','opt_ar'])
    resume=dest/'resume.pth'; start_index=0; total_steps=0; elapsed_prior=0.0
    if resume.exists():
        cp=torch.load(resume,map_location='cpu',weights_only=True)
        if cp['job']!=job or cp['source_files']!=source_files: raise RuntimeError('Resume training protocol mismatch')
        agent.load_state_dict(cp['agent_state_dict'],strict=True)
        for name in optim_names:getattr(agent,name).load_state_dict(cp['optimizers'][name])
        start_index=cp['next_index'];total_steps=cp['total_steps'];elapsed_prior=cp['elapsed_training_seconds']
    initial_state=dest/'initial.pth'
    if not initial_state.exists():save_checkpoint(agent,initial_state,{'job':job,'kind':'initial_random_weights'})
    begin=time.perf_counter()
    atomic_json(dest/'protocol.json',{'job':job,'source_files':source_files,'epochs':2,'instances':len(dataset),
                'checkpoint_selection':'fixed_final_after_all_source_episodes','validation_used_for_selection':False,
                'source_schedule':'sequential_episodes; optimizer persists; no test transitions',
                'rollout_size':8 if mode=='light' else 128,'source_hashes':source_manifest()})
    for index in range(start_index,len(tasks)):
        if (ROOT/'STOP_AFTER_CURRENT_RUN').exists():break
        epoch,r=tasks[index]
        steps=1000 if mode=='light' else int(r.get('original_steps_per_epoch',max(100,max(2000,50*r['n'])//10)))
        config={'problem':problem,'method':method,'instance_path':r['path'],'seed':job['training_seed']+epoch*100000+index,
                'budget_mode':'iterations','budget_value':steps,'rollout_size':8 if mode=='light' else 128,
                'device':'cpu','record_parameter_hash':False,'experiment':'TRAIN','run_id':f'{jid}_{index:05}',
                'instance_id':r['id'],'group':r['group'],'data_sha256':r['sha256'],'epoch':epoch}
        result=run_trial(config,agent=agent)
        val=validate_routes(r['path'],problem,result['routes'])
        if result['status']!='ok' or not val['valid']:
            atomic_json(dest/'failed_episode.json',{'config':config,'result':result,'validation':val})
            raise RuntimeError('Source training episode failed independent validation')
        result['validation']=val
        result['source_episode_index']=index
        atomic_json(ROOT/'logs'/'training'/jid/f'{index:05}.json',result)
        total_steps+=result['iterations']
        elapsed=elapsed_prior+time.perf_counter()-begin
        state={'job_id':jid,'status':'running','completed_episodes':index+1,'total_episodes':len(tasks),
               'total_steps':total_steps,'elapsed_training_seconds':elapsed,'updated_at':timestamp(),
               'current_instance':r['id'],'pid':os.getpid()}
        atomic_json(dest/'progress.json',state)
        # Saving every episode gives exact resumability without losing data.
        payload={'agent_state_dict':agent.state_dict(),'optimizers':{n:getattr(agent,n).state_dict() for n in optim_names},
                 'next_index':index+1,'total_steps':total_steps,'elapsed_training_seconds':elapsed,
                 'source_files':source_files,'job':job}
        tmp=dest/'resume.tmp.pth';torch.save(payload,tmp);os.replace(tmp,resume)
        print(jid,index+1,'/',len(tasks),total_steps,flush=True)
    completed=read_json(dest/'progress.json') if (dest/'progress.json').exists() else {}
    if completed.get('completed_episodes')==len(tasks):
        metadata={'job':job,'source_files':source_files,'global_steps':total_steps,'epochs':2,
                  'selection':'fixed_final','elapsed_training_seconds':completed['elapsed_training_seconds'],
                  'source_protocol_path':str(dest/'protocol.json')}
        digest=save_checkpoint(agent,final,metadata)
        completed.update(status='complete',checkpoint_path=str(final),checkpoint_sha256=digest,completed_at=timestamp())
        atomic_json(done,completed)
        atomic_json(dest/'progress.json',completed)
    return completed

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args()
    train(read_json(args.job))
