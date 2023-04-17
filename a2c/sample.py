from encoder import A2CGPTEncoderConfig, A2CGPTEncoderModel

encoder = A2CGPTEncoderModel(A2CGPTEncoderConfig())
actions, losses = encoder.sample_nograd("Hello", 0, 0, 1, 50, False)
for action in actions:
    print(action)
