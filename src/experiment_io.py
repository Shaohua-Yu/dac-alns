"""Small standard-library utilities for the revision experiment archive."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import time
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]

def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    # Windows readers/antivirus may briefly deny replacement of an existing
    # destination. Keep the old complete document until replacement succeeds.
    for attempt in range(60):
        try:
            os.replace(tmp, path)
            break
        except OSError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==59:
                raise
            time.sleep(min(0.05*(attempt+1),0.5))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def source_manifest():
    files = list((ROOT/'src').glob('*.py')) + list((ROOT/'original_source').glob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted(files)}

def physical_core_slots():
    """One logical processor from each physical core, across Windows groups."""
    if os.name != 'nt':
        return [{'group':0,'mask':1 << c,'logical_cpu':c} for c in range(os.cpu_count() or 1)]
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    n = ctypes.c_ulong(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))
    b = ctypes.create_string_buffer(n.value)
    if not k.GetLogicalProcessorInformationEx(0,b,ctypes.byref(n)):
        raise ctypes.WinError(ctypes.get_last_error())
    data = b.raw[:n.value]
    slots=[]; offset=0
    while offset < len(data):
        relation,size=struct.unpack_from('<II',data,offset)
        if not size: raise RuntimeError('Invalid processor topology record')
        if relation==0:
            count=struct.unpack_from('<H',data,offset+30)[0]
            for j in range(count):
                mask,group=struct.unpack_from('<QH',data,offset+32+j*16)
                if mask:
                    first=mask & -mask
                    slots.append({'group':group,'mask':first,'logical_cpu':first.bit_length()-1})
                    break
        offset+=size
    if not slots: raise RuntimeError('No physical CPU slots discovered')
    return slots

def bind_slot(slot):
    if not slot: return None
    if os.name!='nt':
        os.sched_setaffinity(0,{slot['logical_cpu']})
        return slot
    class GroupAffinity(ctypes.Structure):
        _fields_=[('Mask',ctypes.c_size_t),('Group',ctypes.c_ushort),('Reserved',ctypes.c_ushort*3)]
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentThread.restype=ctypes.c_void_p
    k.SetThreadGroupAffinity.argtypes=[ctypes.c_void_p,ctypes.POINTER(GroupAffinity),ctypes.POINTER(GroupAffinity)]
    affinity=GroupAffinity(int(slot['mask']),int(slot['group']),(ctypes.c_ushort*3)(0,0,0))
    old=GroupAffinity()
    if not k.SetThreadGroupAffinity(k.GetCurrentThread(),ctypes.byref(affinity),ctypes.byref(old)):
        raise ctypes.WinError(ctypes.get_last_error())
    k.GetThreadGroupAffinity.argtypes=[ctypes.c_void_p,ctypes.POINTER(GroupAffinity)]
    actual=GroupAffinity()
    if not k.GetThreadGroupAffinity(k.GetCurrentThread(),ctypes.byref(actual)):
        raise ctypes.WinError(ctypes.get_last_error())
    if actual.Group!=affinity.Group or actual.Mask!=affinity.Mask:
        raise RuntimeError('CPU binding did not take effect')
    return {'group':int(actual.Group),'mask':int(actual.Mask)}

def capture_machine():
    import psutil
    slots=physical_core_slots()
    return {'captured_at':timestamp(),'platform':platform.platform(),'python':platform.python_version(),
            'logical_processors':os.cpu_count(),'physical_cores':len(slots),
            'memory_total_bytes':psutil.virtual_memory().total,'memory_available_bytes':psutil.virtual_memory().available,
            'disk_free_bytes':psutil.disk_usage(str(ROOT)).free,'physical_core_slots':slots,
            'thread_environment':{k:os.getenv(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}

if __name__=='__main__':
    m=capture_machine()
    atomic_json(ROOT/'reports'/'machine.json',m)
    print(json.dumps({k:v for k,v in m.items() if k!='physical_core_slots'},ensure_ascii=False))
    print('groups',sorted({s['group'] for s in m['physical_core_slots']}))
    for g in sorted({s['group'] for s in m['physical_core_slots']}):
        print('bind',bind_slot(next(s for s in m['physical_core_slots'] if s['group']==g)))
