import os, sys, shutil, subprocess
sys.path.insert(0, "ml_stable_diffusion")
from python_coreml_stable_diffusion import torch2coreml
from diffusers import StableDiffusionPipeline
from vocab import convert_vocab
import b2sdk.v2

if __name__ == "__main__":
    # clear models directory
    shutil.rmtree("models")
    os.makedirs("models")
    convert_vocab("models")

    # retrieve pytorch models from huggingface
    pipe, args = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")), {}
    args.compute_unit = "ALL"
    args.attention_implementation = "SPLIT_EINSUM"
    args.o = "models"

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
