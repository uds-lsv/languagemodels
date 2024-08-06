from transformers.trainer import Trainer

class RnnLMTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"], 
            labels=inputs["labels"],
            hidden_state=None,
            pad_id=self.tokenizer.pad_token_id, 
            return_dict=True
        )

        loss = outputs.pop("loss")

        return (loss, outputs) if return_outputs else loss
