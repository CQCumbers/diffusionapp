from tensorflow import keras
import coremltools as ct

text_encoder_fpath = keras.utils.get_file(
    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5",
    file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
)
diffusion_model_fpath = keras.utils.get_file(
    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
    file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
)
decoder_fpath = keras.utils.get_file(
    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
    file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
)
#encoder_fpath = keras.utils.get_file(
#    origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
#    file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
#)

text_encoder = keras.models.load_model(text_encoder_fpath)
ml_encoder = ct.convert(text_encoder, compute_precision=ct.precision.FLOAT16, convert_to="mlprogram")
ml_encoder.save("txt_encoder.mlpackage")

diffusion_model = keras.models.load_weights(diffusion_model_fpath)
ml_unet = ct.convert(diffusion_model, compute_precision=ct.precision.FLOAT16, convert_to="mlprogram")
ml_unet.save("unet.mlpackage")

decoder = keras.models.load_weights(decoder_fpath)
ml_decoder = ct.convert(diffusion_model, compute_precision=ct.precision.FLOAT16, convert_to="mlprogram")
ml_decoder.save("decoder.mlpackage")
