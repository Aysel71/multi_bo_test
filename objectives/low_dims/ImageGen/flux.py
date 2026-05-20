import os
import torch
import random
import fnmatch
import sys
# sys.path.append('../prompt-to-prompt-with-sdxl/')
from ptp.flux_attention_editing_pipeline import FLUXAttentionEditingPipeline
from ptp.utils_img_processing import WarpingTransform
from ptp.processors import *
import gradio as gr
import math
import threading
from PIL import Image
from diffusers import FluxPipeline

# data generating helper functions
def utility(X, opt, device=None, maximize=True):
    """Given X, output corresponding utility (i.e., the latent function)"""
    # y is user score for a choice of X: attention edit parameters,
    
    results = []
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    seed = opt.img_seed
    prompts=opt.prompts
    t_edit = opt.t_edit
    delta=opt.delta
    cross_maps_path=opt.cross_maps_path
    res_blk=opt.res_blk
    attn_blk=opt.attn_blk
    attn_res=opt.attn_res
    edit_type=opt.edit_type
    relative_factor=opt.relative_factor
    blending_alphas=opt.blending_alphas
    pad_crop=opt.pad_crop
    pad_kwargs=opt.pad_kwargs
    sr=opt.sr
    ratio=opt.ratio
    output_path=opt.output_path
    feat_blocks = opt.feat_blocks
    feats = opt.feats
    edit_resnet_uncond = opt.edit_resnet_uncond
    before_softmax = opt.before_softmax
    attn_uncond = opt.attn_uncond
    device=opt.device if device is None else device
    
    if isinstance(prompts, str):
        prompts = [prompts]*2
    
    for i in range(X.shape[0]):
        print(" ")
        print(f"sample---{i}----")
        print(" ")
        inp = X[i]
        pipe = FLUXAttentionEditingPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell",torch_dtype=dtype, use_safetensors=True,).to(device)
        pipe.enable_model_cpu_offload()
        gen = torch.Generator().manual_seed(seed)
        comp_words = get_valid_tokens(prompts[1], pipe.tokenizer)[1]
        
        # Use cross-attention masks
        if opt.use_cross_attention_masks:
            for res in [64,32]:
                if not os.path.exists(os.path.join(cross_maps_path,f"per_word_cross_masks_seed-{seed}_res-{res}.pt")):
                    if "-" in res_blk:
                        blk1 = res_blk.split("-")[0]
                        blk1 = blk1[:-1]
                    elif "down" in res_blk:
                        blk1 = "down"
                    elif "up" in res_blk:
                        blk1 = "up"
                    elif "mid" in res_blk:
                        blk1 = "mid"
                    feat_attn_kwargs_1 = {"edit_type": "no-edit",
                                        "n_self_replace": [t_edit,t_edit+delta],
                                                "path": output_path
                                                }
                    print(" ")
                    print(f"storing cross-attention masks")
                    print(" ")
                    image = pipe(prompts, num_inference_steps=opt.num_inference_steps, guidance_scale=0., max_sequence_length=512, feat_attn_kwargs=feat_attn_kwargs_1, generator=gen, feat_blocks=feat_blocks, feats=feats, before_softmax=before_softmax, attn_uncond=attn_uncond, edit_resnet_uncond=edit_resnet_uncond)
                    cross_maps = create_cross_attention_masks(pipe.controller, prompts, [comp_words]*2, pipe.tokenizer, from_where=[blk1], out_res=res, ratio=ratio)
                    torch.save(cross_maps,os.path.join(cross_maps_path,f"per_word_cross_masks_{seed}_{res}.pt"))
                    del image
            cross_maps = torch.load(os.path.join(cross_maps_path,f"per_word_cross_masks_seed-{seed}_res-{attn_res}.pt"),map_location=device)
        else:
            cross_maps = None
        n_words = len(comp_words)

        if edit_type == "composite":
            shift_pos = [tuple(inp[j:j+2].tolist()) for j in range(0, n_words * 2, 2)]
            offset = n_words * 2
            scale_strengths = inp[offset : offset + n_words].tolist()
            offset += n_words
            equalizer_strengths = inp[offset : offset + n_words].tolist()
            offset += n_words
            rotate_angles = inp[offset : offset + n_words].tolist()
        elif edit_type =="affine":
            n_params_per_word = opt.dim // n_words
            affine_params = [tuple(inp[j:j+n_params_per_word].tolist()) for j in range(0, n_words * n_params_per_word, n_params_per_word)]

        else:
            n_params_per_word = opt.dim
            affine_params = [tuple(inp[j:j+n_params_per_word].tolist()) for j in range(0, n_words * n_params_per_word, n_params_per_word)]

        
        if edit_type == "composite":
            op_params = {
                "shift": {"shift_pos": shift_pos},
                "scale": {"scale_strengths": scale_strengths},
                "reweight": {"equalizer_strengths": equalizer_strengths},
                "rotate": {"rotate_angles": rotate_angles},
            }
            operators = ["shift","scale","reweight","rotate"]
        elif edit_type =="affine":
            op_params = {"affine": {"params": affine_params}}
            operators = None
        else:    
            aff_trans  = WarpingTransform(opt.sr,opt.geometric_model,opt._t, opt._s, opt._alpha, opt._s_alpha,
                opt._t_tps_for_afftps, opt._t_hom, opt._t_tps, opt.tps_grid_size, opt.tps_reg_factor,
                opt.transformation_types, opt._horizontal_flip,
                nbr_perturbations=opt.nbr_perturbations, elastic_parameters=opt.elastic_parameters, 
                sigma_mask=opt.sigma_mask, device=device, seed=seed, use_elastic=opt.use_elastic,blend=opt.blend)
            operators = None
            op_params = {"geometric": {"params": affine_params, "trans":aff_trans}}
            

        edit_blocks = {}
        edit_blocks = attn_blk
        edit_blocks["resnet"] = res_blk.split("-") if "-" in res_blk else [res_blk]

        feat_attn_kwargs = {"edit_type": edit_type,
                                "path": output_path,
                                "edit_blocks": edit_blocks,
                                "per_word_cross_masks": cross_maps,
                                "delta": delta,
                                "place_in_unet": res_blk.split("-")[0] if "-" in res_blk else res_blk,
                                "relative_factor": relative_factor,
                                "blending_alphas": blending_alphas,
                                "pad_crop": pad_crop,
                                "pad_kwargs":pad_kwargs,
                                "sr": sr,
                                "edit_start_step": t_edit,
                                "comp_words": comp_words,
                                # "local_blend_words": [["cat", "dog"], ["cat", "dog"]],
                                "operators": operators,
                                "op_params": op_params,
                                "visualize": opt.visualize,
                                "print_blk": opt.print_blk,
                                "extra_kwargs": opt.extra_kwargs,
                                }
        print(op_params)
        imgs = pipe(prompts, num_inference_steps=opt.num_inference_steps, guidance_scale=0., max_sequence_length=512, feat_attn_kwargs=feat_attn_kwargs, generator=gen, feat_blocks=feat_blocks, feats=feats, before_softmax=before_softmax, attn_uncond=attn_uncond, edit_resnet_uncond=edit_resnet_uncond)['images']
        image = imgs[1]
        image_orig = imgs[0]
        file_n = len(fnmatch.filter(os.listdir(output_path), '*.png'))
        image.save(os.path.join(output_path,f"image_{file_n+1}.png"))
        os.makedirs(os.path.join(output_path,"image_orig"),exist_ok=True)
        if not os.path.exists(os.path.join(output_path,"image_orig",f"image_orig.png")):
            image_orig.save(os.path.join(output_path,"image_orig",f"image_orig.png"))
        results.append(image)
        del pipe, image, image_orig, feat_attn_kwargs
        if cross_maps is not None:
            del cross_maps
        torch.cuda.empty_cache()

    return results
    # rate_image(image)

def unedited_generation(prompt, seed, num_inference_steps, device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=dtype).to(device)

    im = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=[torch.manual_seed(seed)] 
        ).images[0]
    del pipeline
    return im

def optimum(dims):
    return torch.zeros(1).item()

def rate_image(image):
    
    """
    Displays a given image and lets the user rate its quality on a 1–10 scale.
    Returns the numeric rating.
    """
    def submit_rating(rating):
        print(f"User rated this image: {rating}/10")
        return rating

    with gr.Blocks() as demo:
        gr.Markdown("### Rate the image quality (1–10)")
        gr.Image(value=image, label="Generated Image", interactive=False, height=512, width=512)
        rating = gr.Slider(1, 10, step=1, label="Your rating")
        submit = gr.Button("Submit Rating")
        output = gr.Textbox(label="Feedback", interactive=False)

        submit.click(fn=submit_rating, inputs=rating, outputs=output)

    demo.launch(share=True)

def record_choice(images, title=None,MAX_COLS=4, json_input=".json", orig_path=None):
    TOT = len(images)
    labels = [f"Image {i}" for i in range(TOT)]
    result = {"choice": None}
    done = threading.Event()

    if title is None:
        title = "Preferential BO Demo"
        
    def submit_choice(choice_label):
        idx = labels.index(choice_label)
        print(f"User selected image index: {idx}")
        result["choice"] = idx
        done.set()  # signal completion
        return int(idx)
    
    # Function to handle JSON file reading
    def process_raw_text(file_path):
        if not file_path or not os.path.exists(file_path):
            return "Error: Invalid path or file not found."
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read() # Return raw string instead of parsed JSON
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
    with gr.Blocks() as demo:
        gr.HTML(f"<h2><center>{title}</center></h2>") # Centered H1
        gr.Markdown(f"### Choose your preferred image (1 of {TOT})")

        # --- Added Collapsible JSON Section ---
        with gr.Accordion("View Config", open=False):
            json_path_input = gr.Textbox(label="File Path", value=json_input)
            
            text_display = gr.Code(label="Raw Content", language="json", interactive=False)
            
            json_path_input.change(fn=process_raw_text, inputs=json_path_input, outputs=text_display)
            demo.load(fn=process_raw_text, inputs=json_path_input, outputs=text_display)
        # --------------------------------------
        with gr.Accordion("View unedited and Target Images", open=False):
            with gr.Row():
                # Column 1: Handles First Image
                with gr.Column():
                    if orig_path is not None:
                        gr.Image(value=Image.open(orig_path), label="Unedited Image", height=512, width=512, interactive=False, show_fullscreen_button=True)
                    else:
                        gr.Textbox(label="Unedited Image", value="No Image Found", interactive=False)

                # Column 2: Handles Second Image
                with gr.Column():
                    if target is not None:
                        gr.Image(value=target, label="Target Image", height=512, width=512, interactive=False, show_fullscreen_button=True)
                    else:
                        gr.Textbox(label="Target Image", value="No Image Found", interactive=False)
        # --------------------------------------                
        num_rows = math.ceil(TOT / MAX_COLS)
        for r in range(num_rows):
            with gr.Row():
                for i in range(r * MAX_COLS, min((r + 1) * MAX_COLS, TOT)):
                    gr.Image(value=images[i], label=f"Image {i}", interactive=False, show_fullscreen_button=True)


        choice = gr.Radio(labels, label="Select the image number you prefer:")
        submit = gr.Button("Submit Choice")
        output = gr.Number(label="Chosen index (0-based)")
        submit.click(fn=submit_choice, inputs=choice, outputs=output)

    # Run Gradio in a background thread so we can block until user submits
    thread = threading.Thread(target=lambda: demo.launch(share=False, inbrowser=True, prevent_thread_lock=True))
    thread.start()

    # Wait until user makes a choice
    done.wait()

    # Close the app
    demo.close()
    thread.join(timeout=1)

    return result["choice"]

def non_human_choice(images, prompt, scoring_metric="pickscore", opt=None, target=None, device="cuda"):
    """
    Evaluates a list of images against a prompt using a specific metric 
    and returns the index of the highest-scoring image.
    """
    if scoring_metric == "clip":
        from models.rewards import clip_utils
        model = clip_utils.Selector(device)
    elif scoring_metric == "aesthetic":
        from models.rewards import aes_utils
        # Path should be defined or passed; using a default placeholder
        model = aes_utils.Selector(device, opt.aes_model_path)
    elif scoring_metric == "hpsv2":
        from models.rewards import hps_utils
        model = hps_utils.Selector(device, opt.hps_model_path)
    elif scoring_metric == "picscore":
        from models.rewards import pickscore_utils
        model = pickscore_utils.Selector(device)
    elif scoring_metric == "hpsv3":
        from models.rewards import hpsv3_utils
        model = hpsv3_utils.Selector(device)
    elif scoring_metric == "imagereward":
        import ImageReward as RM
        model = RM.load("ImageReward-v1.0")
        model = model.to(device)
    elif scoring_metric == "lpips-clip-ssim":
        from models.rewards import lpips_ssim_clip
        model = lpips_ssim_clip.Selector(device)
    else:
        raise ValueError(f"Unknown metric: {scoring_metric}")

    scores = []
    for img in images:
        if scoring_metric == "imagereward":
            score = model.score(prompt, [img])
        elif scoring_metric == "lpips-clip-ssim":
            target_img = Image.open(opt.target_img_path).convert('RGB')
            score = model.score([target_img],[img])[0]
        else:
            score = model.score([img], prompt)[0]
        
        if torch.is_tensor(score):
            score = score.cpu().item()

        scores.append(score)
    max_score = max(scores)
    best_index = scores.index(max_score)
    return best_index

# def record_choice(image_a, image_b):
#     """
#     Displays two given images and lets the user select which one they prefer.
#     Returns their choice.
#     """
#     def submit_choice(choice):
#         print(f"User selected image: {choice}")
#         return torch.Tensor([int(choice), int(1-choice)])

#     with gr.Blocks() as demo:
#         gr.Markdown("### Choose which image you prefer")
#         with gr.Row():
#             gr.Image(value=image_a, label="Image 0", interactive=False)
#             gr.Image(value=image_b, label="Image 1", interactive=False)

#         choice = gr.Radio([0, 1], label="Which image do you prefer?")
#         submit = gr.Button("Submit Choice")
#         output = gr.Textbox(label="Feedback", interactive=False)

#         submit.click(fn=submit_choice, inputs=choice, outputs=output)

#     demo.launch(share=True)