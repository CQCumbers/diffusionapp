# within torch2coreml.py change
# minimum_deployment_target to ct.target.macOS12
# inputs type to np.float32

import os, sys, types, shutil, functools
import subprocess, b2sdk.v2, torch
from collections import namedtuple
import coremltools as ct
from diffusers import StableDiffusionPipeline
from vocab import convert_vocab

sys.path.insert(0, "ml_stable_diffusion")
from python_coreml_stable_diffusion import torch2coreml
torch.jit.trace = functools.partial(
    torch.jit.trace, strict=False, check_trace=False
)

def get_out_path(args, submodule_name):
    fname = f"{submodule_name}.mlpackage"
    fname = fname.replace("/", "_")
    return os.path.join(args.o, fname)
torch2coreml._get_out_path = get_out_path


if __name__ == "__main__":
    # clear models directory
    upload = len(sys.argv) > 1 and sys.argv[1] == "--upload"
    os.makedirs("models", exist_ok=True)
    convert_vocab("models")

    # retrieve pytorch models from huggingface
    args = types.SimpleNamespace()
    args.model_version = "CompVis/stable-diffusion-v1-4"
    args.attention_implementation = "SPLIT_EINSUM"
    args.compute_unit = "ALL"
    args.chunk_unet = False
    args.latent_h = 0
    args.latent_w = 0
    args.check_output_correctness = True
    args.o = "models"

    pipe = StableDiffusionPipeline.from_pretrained(args.model_version,
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

    # convert models with ml-stable-diffusion
    print("Converting vae_decoder")
    torch2coreml.convert_vae_decoder(pipe, args)
    print("Converting unet")
    torch2coreml.convert_unet(pipe, args)
    print("Converting text_encoder")
    torch2coreml.convert_text_encoder(pipe, args)

    # compile models to mlmodelc
    for model in os.listdir("models"):
        if not model.endswith(".mlpackage"): continue
        subprocess.run(["xcrun", "coremlc", "compile", f"models/{model}", "models"])
        if upload: shutil.rmtree(f"models/{model}")

    # zip and upload models to b2
    if upload:
        zipname = "diffusionapp_models.zip"
        os.remove(zipname)
        subprocess.run(["zip", "-r", zipname, "models"])
        sha1 = subprocess.check_output(["shasum", zipname], text=True).split()

        b2_api = b2sdk.v2.B2Api(b2sdk.v2.InMemoryAccountInfo())
        b2_api.authorize_account("production", os.getenv("B2_ACCOUNT_ID"), os.getenv("B2_ACCOUNT_KEY"))
        bucket = b2_api.get_bucket_by_name("cqcumbers-public-b2")
        bucket.upload_local_file(local_file=zipname, file_name=zipname, sha1_sum=sha1[0])
