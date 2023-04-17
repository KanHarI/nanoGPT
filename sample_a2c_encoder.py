from a2c import A2CGPTEncoderConfig, A2CGPTEncoderModel

encoder = A2CGPTEncoderModel(A2CGPTEncoderConfig())
print(encoder.sample_nograd("Hello world!", 0, 0, 1))
