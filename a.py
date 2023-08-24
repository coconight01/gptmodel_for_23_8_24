from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
model_path="/shared_disk/TinyStories/modela/mymodel"#改为mymodel的路径
model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
while True:
    line = input()
    prompt = line
    print("start")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output=model.generate(input_ids,max_length=50,num_beams=1)

    output_text=tokenizer.decode(output[0],skip_special_tokens=True)

    print(output_text)