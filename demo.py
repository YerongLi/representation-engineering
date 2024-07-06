from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
repe_pipeline_registry()

# ... initializing model and tokenizer ....
model_name = "microsoft/phi-2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

control_kwargs={}
rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)
