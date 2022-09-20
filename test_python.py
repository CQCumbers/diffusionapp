# comment out line 360-362 _get_proxy_and_spec in coremltools/models/model.py
# and on line 358 replace _save_spec with self._spec = model
import torch, os
from diffusers import StableDiffusionPipeline
access_token = os.getenv('HUGGINGFACE_TOKEN')
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
    revision="fp16", dtype=torch.bfloat16, use_auth_token=access_token)

import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from pathlib import Path
import torch as th
import diffusers

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

def convert_text_encoder(text_encoder, outname):
    import transformers
    transformers.models.clip.modeling_clip.CLIPTextTransformer.attention_mask = transformers.models.clip.modeling_clip.CLIPTextTransformer._build_causal_attention_mask(None, 1, 77, th.float)
    def _fake_build_causal_mask(self, *args, **kwargs):
        return self.attention_mask
    transformers.models.clip.modeling_clip.CLIPTextTransformer._build_causal_attention_mask = _fake_build_causal_mask
    f_trace = th.jit.trace(CLIPUndictifier(text_encoder), (th.zeros(1, 77, dtype=th.long)), strict=False, check_trace=False)

    f_coreml = ct.convert(f_trace,
       inputs=[ct.TensorType(shape=(1, 77))],
       convert_to="milinternal",
       compute_precision=ct.precision.FLOAT16,
       skip_model_load=True)
    f_coreml = ct.convert(f_coreml, convert_to="neuralnetwork")
    f_coreml = quantization_utils.quantize_weights(f_coreml, nbits=16)
    f_coreml.save(outname)

def convert_decoder(decoder, outname):    
    f_trace = th.jit.trace(decoder, (th.zeros(1, 4, 64, 64)), strict=False, check_trace=False)

    f_coreml = ct.convert(f_trace, 
       inputs=[ct.TensorType(shape=(1, 4, 64, 64))],
       convert_to="milinternal",
       compute_precision=ct.precision.FLOAT16,
       skip_model_load=True)
    f_coreml = ct.convert(f_coreml, convert_to="neuralnetwork")
    f_coreml = quantization_utils.quantize_weights(f_coreml, nbits=16)
    f_coreml.save(outname)

def convert_post_quant_conv(layer, outname):
    f_trace = th.jit.trace(layer, (th.zeros(1, 4, 64, 64)), strict=False, check_trace=False)

    f_coreml = ct.convert(f_trace, 
        inputs=[ct.TensorType(shape=(1, 4, 64, 64))],
        convert_to="milinternal",
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True)
    f_coreml = ct.convert(f_coreml, convert_to="neuralnetwork")
    f_coreml = quantization_utils.quantize_weights(f_coreml, nbits=16)
    f_coreml.save(outname)

def convert_unet(f, out_name):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
    import coremltools.converters.mil.frontend.torch.ops as cml_ops
    def unsliced_attention(self, query, key, value, _sequence_length, _dim):
        attn = (torch.einsum("b i d, b j d -> b i j", query, key) * self.scale).softmax(dim=-1)
        attn = torch.einsum("b i j, b j d -> b i d", attn, value)
        return self.reshape_batch_dim_to_heads(attn)
    diffusers.models.attention.CrossAttention._attention = unsliced_attention
    orig_einsum = th.einsum
    def fake_einsum(a, b, c):
        if a == 'b i d, b j d -> b i j': return th.bmm(b, c.permute(0, 2, 1))
        if a == 'b i j, b j d -> b i d': return th.bmm(b, c)
        raise ValueError(f"unsupported einsum {a} on {b.shape} {c.shape}")
    th.einsum = fake_einsum
    if "broadcast_to" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["broadcast_to"]
    @register_torch_op
    def broadcast_to(context, node): return cml_ops.expand(context, node)
    if "gelu" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["gelu"]
    @register_torch_op
    def gelu(context, node): context.add(mb.gelu(x=context[node.inputs[0]], name=node.name))
    
    print("tracing")
    f_trace = th.jit.trace(Undictifier(f), (th.zeros(2, 4, 64, 64), th.zeros(1), th.zeros(2, 77, 768)), strict=False, check_trace=False)
    print("converting")
    f_coreml = ct.convert(f_trace, 
       inputs=[ct.TensorType(shape=(2, 4, 64, 64)), ct.TensorType(shape=(1,)), ct.TensorType(shape=(2, 77, 768))],
       convert_to="milinternal",
       compute_precision=ct.precision.FLOAT16,
       skip_model_load=True)
    f_coreml = ct.convert(f_coreml, convert_to="neuralnetwork")
    f_coreml = quantization_utils.quantize_weights(f_coreml, nbits=16)
    f_coreml.save(f"{out_name}")
    th.einsum = orig_einsum
    
class UNetWrapper:
    def __init__(self, f, out_name="unet.mlmodel"):
        self.in_channels = f.in_channels
        if not Path(out_name).exists():
            print("generating coreml model"); convert_unet(f, out_name); print("saved")
        # not only does ANE take forever to load because it recompiles each time - it then doesn't work!
        # and NSLocalizedDescription = "Error computing NN outputs."; is not helpful... GPU it is
        print("loading saved coreml model"); f_coreml_fp16 = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_ONLY); print("loaded")
        self.f = f_coreml_fp16

    def __call__(self, sample, timestep, encoder_hidden_states):
        args = {"sample_1": sample.numpy(), "timestep": th.tensor([timestep], dtype=th.int32).numpy(), "input_35": encoder_hidden_states.numpy()}
        for v in self.f.predict(args).values():
            return diffusers.models.unet_2d_condition.UNet2DConditionOutput(sample=th.tensor(v, dtype=th.float32))

class TextEncoderWrapper:
    def __init__(self, f, out_name="text_encoder.mlmodel"):
        if not Path(out_name).exists():
            print("generating coreml model"); convert_text_encoder(f, out_name); print("saved")
        print("loading saved coreml model"); self.f = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_AND_GPU); print("loaded")
    
    def __call__(self, input):
        args = args = {"input_ids_1": input.float().numpy()}
        for v in self.f.predict(args).values():
            return (th.tensor(v, dtype=th.float32),)

class DecoderWrapper:
    def __init__(self, f, out_name="vae_decoder.mlmodel"):
        if not Path(out_name).exists():
            print("generating coreml model"); convert_decoder(f, out_name); print("saved")
        print("loading saved coreml model"); f_coreml_fp16 = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_ONLY); print("loaded")
        self.f = f_coreml_fp16
    
    def __call__(self, input):
        args = args = {"z": input.numpy()}
        for v in self.f.predict(args).values():
            return th.tensor(v, dtype=th.float32)

class PostQuantConvWrapper:
    def __init__(self, f, out_name="post_quant_conv.mlmodel"):
        if not Path(out_name).exists():
            print("generating coreml model"); convert_post_quant_conv(f, out_name); print("saved")
        print("loading saved coreml model"); f_coreml_fp16 = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_ONLY); print("loaded")
        self.f = f_coreml_fp16
    
    def __call__(self, input):
        args = {"input": input.numpy()}
        for v in self.f.predict(args).values():
            return th.tensor(v, dtype=th.float32)

class VAEWrapper:
    def __init__(self, decoder, post_quant_conv):
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv

    def decode(self, input):
        quant = self.post_quant_conv(input)
        dec = self.decoder(quant)

        return diffusers.models.vae.DecoderOutput(sample=dec)

pipe.text_encoder = TextEncoderWrapper(pipe.text_encoder)
pipe.unet = UNetWrapper(pipe.unet)
pipe.vae = VAEWrapper(
            DecoderWrapper(pipe.vae.decoder), 
            PostQuantConvWrapper(pipe.vae.post_quant_conv)) 

pipe.safety_checker = lambda images, **kwargs: (images, False)
generator = torch.Generator("cpu").manual_seed(123)
prompt = "discovering ancient ruins, concept art by JaeCheol Park"
image = pipe(prompt, num_inference_steps=21, generator=generator).images[0]
image.save("ruins.png")
