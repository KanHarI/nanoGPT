from a2c.gpt_auto_tokenizer import GPTAutoTokenizer, GPTAutoTokenizerConfig

encoder = GPTAutoTokenizer(GPTAutoTokenizerConfig())
actions, losses = encoder.sample_vocab_to_latent([1, 2, 3], 0, 0)
for action in actions:
    print(action)
