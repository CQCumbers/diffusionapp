# from https://gist.github.com/madebyollin/86b9596ffa4ab0fa7674a16ca2aeab3d
# on converters/mil/front/torch.ops.py line 5096 add:
# beta = mb.cast(x=beta, dtype="fp32")
import os, sys, shutil, subprocess
import diffusers, b2sdk.v2
from diffusers import StableDiffusionPipeline

import coremltools as ct
import torch as th
import numpy as np


class Undictifier(th.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, *args, **kwargs):
        return self.m(*args, **kwargs)["sample"]

class CLIPUndictifier(th.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, *args, **kwargs):
        return self.m(*args, **kwargs)[0]

class QuantDecoder(th.nn.Module):
    def __init__(self, q, d):
        super().__init__()
        self.q = q
        self.d = d
    def forward(self, inp):
        tmp = self.q(inp)
        return self.d(tmp)


def compile_mlmodel(filename):
    print(f"Compiling {filename} to models")
    subprocess.run(["xcrun", "coremlc", "compile", filename, "models"])


def convert_text_encoder(text_encoder, out_name):
    print("Generating text encoder model")
    import transformers

    def _fake_build_causal_mask(self, *args, **kwargs):
        return self.attention_mask

    transformers.models.clip.modeling_clip.CLIPTextTransformer.attention_mask = (
        transformers.models.clip.modeling_clip.CLIPTextTransformer._build_causal_attention_mask(None, 1, 77, th.float))
    transformers.models.clip.modeling_clip.CLIPTextTransformer._build_causal_attention_mask = _fake_build_causal_mask

    f_trace = th.jit.trace(CLIPUndictifier(text_encoder),
        (th.zeros(1, 77, dtype=th.long)), strict=False, check_trace=False)
    f_coreml = ct.convert(f_trace,
        inputs=[ct.TensorType(shape=(1, 77))],
        convert_to="milinternal",
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True)
    f_coreml = ct.convert(f_coreml, convert_to="mlprogram")

    spec = f_coreml.get_spec()
    ct.utils.rename_feature(spec, 'input_ids_1', 'in_ids')
    ct.utils.rename_feature(spec, 'last_hidden_state', 'out_embeds')
    ct.models.utils.save_spec(spec, out_name, weights_dir=f_coreml.weights_dir)


def convert_decoder(decoder, quant, out_name):
    print("Generating decoder model")

    # replace baddbmm beta with float

    f_trace = th.jit.trace(QuantDecoder(quant, decoder),
        (th.zeros(1, 4, 64, 64, dtype=th.float32)), strict=False, check_trace=False)
    f_coreml = ct.convert(f_trace, 
        inputs=[ct.TensorType(shape=(1, 4, 64, 64))],
        convert_to="milinternal",
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True)
    f_coreml = ct.convert(f_coreml, convert_to="mlprogram")

    spec = f_coreml.get_spec()
    ct.utils.rename_feature(spec, 'z', 'in_z')
    ct.utils.rename_feature(spec, 'var_665', 'out_image')
    ct.models.utils.save_spec(spec, out_name, weights_dir=f_coreml.weights_dir)


def convert_unet(f, out_name):
    print("Generating diffuser model")
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
    import coremltools.converters.mil.frontend.torch.ops as cml_ops

    def unsliced_attention(self, query, key, value):
        attn = (th.einsum("b i d, b j d -> b i j", query, key) * self.scale).softmax(dim=-1)
        attn = th.einsum("b i j, b j d -> b i d", attn, value)
        return self.reshape_batch_dim_to_heads(attn)

    def fake_einsum(a, b, c):
        if a == 'b i d, b j d -> b i j': return th.bmm(b, c.permute(0, 2, 1))
        if a == 'b i j, b j d -> b i d': return th.bmm(b, c)
        raise ValueError(f"unsupported einsum {a} on {b.shape} {c.shape}")

    diffusers.models.attention.CrossAttention._attention = unsliced_attention
    orig_einsum = th.einsum
    th.einsum = fake_einsum

    if "broadcast_to" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["broadcast_to"]
    @register_torch_op
    def broadcast_to(context, node): return cml_ops.expand(context, node)

    if "gelu" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["gelu"]
    @register_torch_op
    def gelu(context, node): context.add(mb.gelu(x=context[node.inputs[0]], name=node.name))
    
    f_trace = th.jit.trace(Undictifier(f),
        (th.zeros(2, 4, 64, 64), th.zeros(1), th.zeros(2, 77, 768)), strict=False, check_trace=False)
    f_coreml = ct.convert(f_trace, 
       inputs=[ct.TensorType(shape=(2, 4, 64, 64)), ct.TensorType(shape=(1,)), ct.TensorType(shape=(2, 77, 768))],
       convert_to="milinternal",
       compute_precision=ct.precision.FLOAT16,
       skip_model_load=True)
    f_coreml = ct.convert(f_coreml, convert_to="mlprogram")
    th.einsum = orig_einsum

    spec = f_coreml.get_spec()
    ct.utils.rename_feature(spec, 'sample', 'in_latents')
    ct.utils.rename_feature(spec, 'timestep', 'in_timestep')
    ct.utils.rename_feature(spec, 'input_35', 'in_embeds')
    ct.utils.rename_feature(spec, 'var_5500', 'out_preds')
    ct.models.utils.save_spec(spec, out_name, weights_dir=f_coreml.weights_dir)


class UNetWrapper:
    def __init__(self, f, out_name="diffuser.mlpackage"):
        self.in_channels = f.in_channels
        if not os.path.exists(out_name):
            convert_unet(f, out_name)
        print("Loading diffuser model")
        compile_mlmodel(out_name)
        self.f = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        print("Model loaded")

    def __call__(self, sample, timestep, encoder_hidden_states):
        args = {
            "in_embeds": sample.numpy(),
            "in_timestep": th.tensor([timestep], dtype=th.int32).numpy(),
            "in_latents": encoder_hidden_states.numpy()
        }
        print(f'Calling with args {args}')
        for v in self.f.predict(args).values():
            print(f'Return value {v}')
            return diffusers.models.unet_2d_condition.UNet2DConditionOutput(sample=th.tensor(v, dtype=th.float32))

class TextEncoderWrapper:
    def __init__(self, f, out_name="txt_encoder.mlpackage"):
        if not os.path.exists(out_name):
            convert_text_encoder(f, out_name)
        print("Loading saved text encoder model")
        compile_mlmodel(out_name)
        self.f = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        print("Model loaded")
    
    def __call__(self, input):
        args = {"in_ids": input.float().numpy()}
        for v in self.f.predict(args).values():
            return (th.tensor(v, dtype=th.float32),)

class DecoderWrapper:
    def __init__(self, d, q, out_name="decoder.mlpackage"):
        if not os.path.exists(out_name):
            convert_decoder(d, q, out_name)
        print("Loading saved decoder model")
        compile_mlmodel(out_name)
        self.f = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        print("Model loaded")
    
    def __call__(self, input):
        args = {"in_z": input.numpy()}
        for v in self.f.predict(args).values():
            return th.tensor(v, dtype=th.float32)

class VAEWrapper:
    def __init__(self, decoder):
        self.decoder = decoder

    def decode(self, input):
        dec = self.decoder(input)
        return diffusers.models.vae.DecoderOutput(sample=dec)


if __name__ == "__main__":
    # clear models directory
    shutil.rmtree("models")
    os.makedirs("models")

    # retrieve pytorch models from huggingface
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
        revision="fp16", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

    # replace components with coreml models
    pipe.text_encoder = TextEncoderWrapper(pipe.text_encoder)
    pipe.unet = UNetWrapper(pipe.unet)
    pipe.vae = VAEWrapper(DecoderWrapper(pipe.vae.decoder, pipe.vae.post_quant_conv))
    pipe.safety_checker = lambda images, **kwargs: (images, False)

    # zip and upload models to b2
    if len(sys.argv) > 1 and sys.argv[1] == "--upload":
        zipname = "diffusionapp_models.zip"
        os.remove(zipname)
        subprocess.run(["zip", "-r", zipname, "models"])
        sha1 = subprocess.check_output(["shasum", zipname], text=True).split()

        b2_api = b2sdk.v2.B2Api(b2sdk.v2.InMemoryAccountInfo())
        b2_api.authorize_account("production", os.getenv("B2_ACCOUNT_ID"), os.getenv("B2_ACCOUNT_KEY"))
        bucket = b2_api.get_bucket_by_name("cqcumbers-public-b2")
        bucket.upload_local_file(local_file=zipname, file_name=zipname, sha1_sum=sha1[0])

    generator = th.Generator("cpu").manual_seed(123)
    #with open("noise.bin", "rb") as f:
    #    data = np.fromfile(f, dtype=np.float32)
    #array = np.reshape(data, [1, 4, 64, 64])
    #loaded_noise = th.from_numpy(array)
    prompt = "discovering ancient ruins, concept art by JaeCheol Park"
    image = pipe(prompt, num_inference_steps=21, generator=generator).images[0]
    #image = pipe(prompt, latents=loaded_noise, num_inference_steps=21).images[0]
    image.save("ruins.png")
