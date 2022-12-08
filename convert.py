import os, sys, types, shutil, functools
import subprocess, b2sdk.v2, torch
from diffusers import StableDiffusionPipeline
from vocab import convert_vocab

sys.path.insert(0, "ml_stable_diffusion")
from python_coreml_stable_diffusion import torch2coreml
torch.jit.trace = functools.partial(
    torch.jit.trace, strict=False, check_trace=False
)


if __name__ == "__main__":
    # clear models directory
    shutil.rmtree("models")
    os.makedirs("models")
    convert_vocab("models")

    # retrieve pytorch models from huggingface
    args = types.SimpleNamespace()
    args.model_version = "CompVis/stable-diffusion-v1-4"
    args.attention_implementation = "SPLIT_EINSUM"
    args.latent_w = 512
    args.latent_h = 512
    args.compute_unit = "ALL"
    args.o = "models"

    pipe = StableDiffusionPipeline.from_pretrained(args.model_version,
        revision="fp16", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

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
        shutil.rmtree(model)

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
