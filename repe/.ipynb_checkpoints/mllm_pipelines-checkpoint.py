from transformers import AutoModel, AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY
from .mllm_rep_reading_pipeline import MLLMRepReadingPipeline
from .mllm_rep_control_pipeline import MLLMRepControlPipeline

def mllm_repe_pipeline_registry():
    PIPELINE_REGISTRY.register_pipeline(
        "mllm-rep-reading",
        pipeline_class=MLLMRepReadingPipeline,
        pt_model=AutoModel,
    )

    PIPELINE_REGISTRY.register_pipeline(
        "mllm-rep-control",
        pipeline_class=MLLMRepControlPipeline,
        pt_model=AutoModelForCausalLM,
    )


