from lm_eval.models.huggingface import HFLM

class InmemoryPeftLM(HFLM):
    """
    LM class for lm_eval that allows for in-memory PEFT models.
    Without this, lm_eval will load the model from scratch every time,
    which we don't need since we already have the model loaded in memory
    during training.
    
    Checked that it produces the same results on an HF model as the CLI
    lm_eval command.
    """
    def __init__(self, peft_model, tokenizer, model_id, device, batch_size):
        self._model = peft_model
        self.tokenizer = tokenizer
        super().__init__(
            pretrained=model_id,
            device=device,
            batch_size=batch_size,
        )

    def _create_model(self, pretrained: str, **kwargs):
        return self._model

    def _get_tokenizer(self, pretrained: str, **kwargs):
        return self.tokenizer