from tqdm import tqdm
from transformers import MistralForCausalLM, LlamaTokenizerFast

from openchat_evaluate import OpenChatDataset
from src.evaluator import GSM8KEvaluator

tokenizer = LlamaTokenizerFast.from_pretrained('/nas-wulanchabu/hongzhan.chz/models/openchat/')
model = MistralForCausalLM.from_pretrained('/nas-wulanchabu/hongzhan.chz/models/openchat/')
model.cuda()

dataset = OpenChatDataset('data/GSM8K/test.json', task='GSM8K')
evaluator = GSM8KEvaluator()
for data in tqdm(dataset):
    print('------------------------------------')
    input_text = data['instruction']
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids.to(next(model.parameters()).device), max_length=512)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    evaluator.forward(generated_text, data['label'])
    print(data['instruction'] + generated_text)
    print("Acc: ", evaluator.accuracy, "Miss: ", evaluator.miss)
