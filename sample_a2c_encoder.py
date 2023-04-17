from a2c import A2CGPTEncoderConfig, A2CGPTEncoderModel

encoder = A2CGPTEncoderModel(A2CGPTEncoderConfig())
actions, losses = encoder.sample_nograd("Hello", 0, 0, 1)
for action in actions:
    print(action)
