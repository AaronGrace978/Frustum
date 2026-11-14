"""Reproducibility fingerprints. These establish identity, not model correctness."""
import hashlib,json,os,platform,sys
from pathlib import Path
import torch

def tensor_sha256(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def file_sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def provenance(model_root):
    folder=Path(__file__).parent
    scripts=['frusium_loader.py','benchmark_layers.py','expert_correctness.py','streamed_moe.py','sparse_rows.py','block_primitives.py','layer0_block.py','reproducibility.py']
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        environment={k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUDA_LAUNCH_BLOCKING','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']},
        source_sha256={n:file_sha256(folder/n) for n in scripts},
        checkpoint_metadata_sha256={n:file_sha256(Path(model_root)/n) for n in ['config.json','model.safetensors.index.json']})

def cache_sha256(loader):
    return {name:tensor_sha256(t) for name,t in sorted(loader.cache.items())}

def compare_reports(a,b):
    checks={
        'same_source':a['provenance']['source_sha256']==b['provenance']['source_sha256'],
        'same_checkpoint_metadata':a['provenance']['checkpoint_metadata_sha256']==b['provenance']['checkpoint_metadata_sha256'],
        'same_stage_fingerprints':a['full_prefill_trace']==b['full_prefill_trace'],
        'same_resident_dense_weights':a['resident_dense_sha256']==b['resident_dense_sha256'],
        'same_router_weights':a['resident_router_sha256']==b['resident_router_sha256'],
        'same_output':a['output_sha256']==b['output_sha256'],
        'same_experts':a['full_prefill_MoE_reference']['selected_experts']==b['full_prefill_MoE_reference']['selected_experts'],
        'same_routing_weights':a['full_prefill_MoE_reference']['routing_weights']==b['full_prefill_MoE_reference']['routing_weights']}
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks)
