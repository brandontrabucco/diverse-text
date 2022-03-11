from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
from datasets import load_dataset
import json
import tqdm


NUM_SAMPLES = 32
MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = 150
DECODE_KWARGS = dict(skip_special_tokens=True)


BEAM_KWARGS = dict(
    no_repeat_ngram_size=2,
    num_beams=NUM_SAMPLES,
    early_stopping=True,
    num_return_sequences=NUM_SAMPLES)


PURE_KWARGS = dict(
    do_sample=True,
    top_k=0,
    temperature=0.7,
    num_return_sequences=NUM_SAMPLES)


TOP_K_KWARGS = dict(
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=1.0,
    num_return_sequences=NUM_SAMPLES)


TOP_P_KWARGS = dict(
    do_sample=True,
    top_k=0,
    top_p=0.9,
    temperature=1.0,
    num_return_sequences=NUM_SAMPLES)


TOP_KP_KWARGS = dict(
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=1.0,
    num_return_sequences=NUM_SAMPLES)


if __name__ == "__main__":

    dataset = load_dataset('wikitext', 'wikitext-103-v1')

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id).cuda()

    all_samples = []

    for example in tqdm.tqdm(dataset["test"]):

        if len(example['text'].strip()) == 0:
            continue  # the dataset contains empty strings

        input_ids = tokenizer.encode(
            example['text'], return_tensors='pt').cuda()

        if input_ids.shape[1] < MIN_SEQUENCE_LENGTH:
            continue  # the dataset contains small strings

        if input_ids.shape[1] > MAX_SEQUENCE_LENGTH:
            continue  # the dataset contains large strings

        input_ids = input_ids[:, :input_ids.shape[1] // 2]

        greedy_output = model.generate(
            input_ids, max_length=MAX_SEQUENCE_LENGTH)

        greedy_decode = [tokenizer.decode(
            greedy_output[0], **DECODE_KWARGS)]

        samples = dict(ground_truth=example['text'],
                       greedy=greedy_decode)

        for name, kwargs in (('beam', BEAM_KWARGS),
                             ('pure', PURE_KWARGS),
                             ('top_k', TOP_K_KWARGS),
                             ('top_p', TOP_P_KWARGS),
                             ('top_kp', TOP_KP_KWARGS)):

            set_seed(0)
            output = model.generate(
                input_ids, max_length=MAX_SEQUENCE_LENGTH, **kwargs)

            samples[name] = [tokenizer.decode(output[i], **DECODE_KWARGS)
                             for i in range(len(output))]

        all_samples.append(samples)
        with open("samples.json", "w") as f:
            json.dump(all_samples, f, indent=4)

