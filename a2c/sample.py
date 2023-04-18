from a2c.gpt_auto_tokenizer import GPTAutoTokenizer, GPTAutoTokenizerConfig

encoder = GPTAutoTokenizer(GPTAutoTokenizerConfig())
actions = encoder.sample_vocab_to_latent([1, 2, 3], 0, 0)
for action in actions:
    print(action)
non_skip_actions = [action for action in actions if not action.shift]
x = 1
