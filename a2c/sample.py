from encoder import A2CGPTEncoderConfig, A2CGPTEncoderModel

encoder = A2CGPTEncoderModel(A2CGPTEncoderConfig())
actions, losses = encoder.sample_nograd("Hello", 0, 0, 1, 50, True)
for action in actions:
    print(action)
restored, losses2 = encoder.decode_nograd(actions, 1, 50, True)
