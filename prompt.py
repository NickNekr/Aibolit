import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Generator:
    prefix = "<s>Категория: Травматология и ортопедия --> Вопрос: "
    suffix = " ==> Ответ: "

    def __init__(self, model_name_or_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model.to("cuda")

    def set_prefix_suffix(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix

    def generate_text(
        self,
        prompt,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    ):
        input_ids = self.tokenizer.encode(
            self.prefix + prompt + self.suffix, return_tensors="pt"
        ).to("cuda")

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,

        )

        generated_text = self.tokenizer.decode(
            output[0], clean_up_tokenization_spaces=True
        )
        return generated_text


generator = GPT2Generator("./aibolit")
prompt = "Подскажите, пожалуйста, вреден ли футбол для здоровья?"
generated_text = generator.generate_text(prompt)
print(generated_text)
