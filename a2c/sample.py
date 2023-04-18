from a2c.gpt_auto_tokenizer import GPTAutoTokenizer, GPTAutoTokenizerConfig

encoder = GPTAutoTokenizer(GPTAutoTokenizerConfig())
actions = encoder.sample_vocab_to_latent([1, 2, 3], 0, 0)
for action in actions:
    print(action)
non_skip_actions = [action for action in actions if not action.shift]
restored = encoder.sample_latent_to_vocab([action.latent for action in non_skip_actions], 0, 0)
for action in restored:
    print(action)