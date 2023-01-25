import torch
from transformers import RobertaTokenizer, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaAdapterModel, AutoTokenizer

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        if "model_args" in model_kargs:
            self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if "model_args" in model_kargs:
            if self.model_args.do_mlm:
                self.lm_head = RobertaLMHead(config)


# Load from pretrained base-model and pretrained adapters
model = RobertaForCL.from_pretrained("roberta-large")
adapter_name = model.roberta.load_adapter("sts-b", model_name="roberta-large", config="houlsby")
model.roberta.set_active_adapters(adapter_name)
p = model.parameters()
print(next(p_ for i, p_ in enumerate(p) if i == 16))

# Save
model.save_pretrained("latest-try")
#model.roberta.save_adapter("latest-try", "sts-b")

# Load from checkpoint
model1 = RobertaForCL.from_pretrained("latest-try")
p1 = model1.parameters()
print(next(p_ for i, p_ in enumerate(p1) if i == 16))
