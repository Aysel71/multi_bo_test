import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pyrallis
import socket

def set_all_seeds(seed):
    """
    Sets the random seed for numpy, random, and pytorch (CPU and CUDA).
    Also sets environment variable for Python hash seed and CUDNN for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior with CUDA operations (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def make_plot_path(opt):
    if opt.multi:
        ff = f"multiwise"
    else:
        ff = f"pairwise"
    if opt.logit:
        ff = ff + '_Logit'
    else:
        ff = ff + '_Probit'
    for algos in opt.acf_algos:
        ff = ff + "_" + f"{algos}"
    ff = ff + "_" + f"q{opt.q}_NUM-TRIALS-{opt.num_trials}_NUM-BATCHES-{opt.num_batches}_NUM-RESTARTS-{opt.num_restarts}_RAW-SAMPLES-{opt.raw_samples}"
    return ff

def plot_results(utility, optimum, best_vals, opt):
    # print(np.vstack(best_vals['TAFR-qEI']))
    plt.rcParams.update({"font.size": 14})

    def ci(y):
        return 1.96 * y.std(axis=0) / np.sqrt(y.shape[0])
    
    iters = list(range(opt.num_batches + 1))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(
        iters,
        [optimum] * len(iters),
        label="Optimal Function Value",
        color="black",
        linewidth=1.5,
    )

    for algo in opt.acf_algos:
        ys = np.vstack(best_vals[algo])
        val = ys.max(axis=0)
        ax.plot(iters, val, label=algo, linewidth=1.5)
        ax.fill_between(iters, val - ci(ys), val + ci(ys), alpha=0.2)
        # ax.errorbar(
        #     iters, val, yerr=ci(ys), label=algo, linewidth=1.5
        # )

    ax.set(
        xlabel=f"Number of queries (q = {opt.num_batches}, num_comparisons = {opt.q})",
        ylabel="Best observed value",
        title=f"{opt.dim}-dim {opt.obj_name}",
    )
    ax.legend(loc="best")

    fig.savefig(os.path.join(opt.output_path,opt.plot_path))
   

def save_results(data, models, best_vals, n_trial, seed, opt):
    res = []
    for algo in opt.acf_algos:
        acf_path = os.path.join(opt.output_path,algo)
        os.makedirs(acf_path, exist_ok=True)
        if opt.obj_name in opt.image_models:  # was: == "sdxl" — flux/sd3/pixart were silently skipped
            for i in [n_trial]: #range(opt.num_trials):
                for j, img in enumerate(best_vals[algo][i]):
                    img.save(os.path.join(acf_path,f"best_image_trial-{i}_batch-{j}.png"))
                    best_vals[algo][i][j] = os.path.join(acf_path,f"best_image_trial-{i}_batch-{j}.png")
                res.append(img)
            # for k, img in enumerate(data[algo][1]):
            #     img.save(os.path.join(acf_path,f"data_image-{k}.png"))
            data[algo] = data[algo][:1] + data[algo][2:]
                
        torch.save(opt.ref_points, os.path.join(acf_path,f"ref_points-algo_{algo}-trial_{n_trial}.pt"))
        opt.ref_points_path = os.path.join(acf_path,f"ref_points.pt")
        config = pyrallis.dump(opt)
        torch.save({
            "acf_algo": algo,
        "data": data[algo],
        "models": models[algo].state_dict(),
        "best_vals": best_vals[algo],
        "trial": n_trial,
        "seed": seed,
        "config": config,
    }, os.path.join(acf_path,f"exp-data-algo_{algo}-trial_{n_trial}.pt"))
        with open(os.path.join(acf_path, f"config-algo_{algo}-trial_{n_trial}.json"), 'w') as f:
            f.write(config)
        print(" ")
        print(f"Results are saved at: {acf_path} for ALGO = {algo}")
    # save_model(models[algo], f"gp_model_trial{n_trial}.pth")
    return res

def save_config(algo, n_trial, seed, opt):
    acf_path = os.path.join(opt.output_path,algo)
    os.makedirs(acf_path, exist_ok=True)
    config = pyrallis.dump(opt)
    save_path = os.path.join(acf_path, f"config-algo_{algo}-trial_{n_trial}.json")
    with open(save_path, 'w') as f:
        f.write(config)
    return save_path