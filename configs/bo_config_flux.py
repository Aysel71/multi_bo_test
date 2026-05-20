from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import torch

@dataclass
class MultiBOConfig_FLUX:
    """Configuration for Multiwise/Pairwise Bayesian Optimization experiments."""

    # Experiment setup
    num_trials: int = 1
    num_batches: int = 20

    # Problem setup
    dim: int = 24 #composite = 10, affine = 14, geo(a) = 48 geo(b) = 24 (grid=3) = 38 (grid=4)
    num_initial_samples: int = 20   # number of datapoints
    m: int = 50  # number of comparisons
    T: int = 3 # num. of choices in choice set
    lim: List[int] = field(default_factory=lambda: [-1,1])
    init_sobol: bool = True
    load_prefit_model: bool = False
    prefit_model_data_path: str = ""

    # objective function
    mode: str = 'low-synthetic'
    obj_name: str = 'x_squared'
    json_input: str = '.json'

    # Acquisition parameters
    num_restarts: int = 50
    raw_samples: int = 4096
    q: int = 2       # number of points per query
    q_comp: int = 1  # number of comparisons per query
    acf_algos: List[str] = field(
        default_factory=lambda: ["EUBO", "qEI", "2s-qEI", "TAFR-qEI", "2s-TAF-qEI"]
    )
    acf: str = ''
    prev_winner: bool = False
    convert_to_pair: bool = False
    
    # UCB parameters
    beta: float = 5.0

    # TAF parameters
    num_population_models: int = 2
    num_population_data: int = 10
    m_pop: int = 10
    increment: bool = False
    d1: float = 2
    d2: float = 0.2
    rho: float = 0.1

    # Two-Step ACF
    num_fantasies: int = 2
    fantasy_trials: int = 20
    ref_points: torch.Tensor = None
    ref_points_path: str = None
    num_ref_points: int = 10
    
    # Manifold ACF
    energy_threshold: float = 0.9
    dbs_eps: float = 0.15
    spectral_ratio: float = 2.0
    d_max: int = 20
    d_min: int = 1
    scale_tol: float = 0.1
    pert_scale: float = 1.5
    dbs_mode = "energy" # spectral, energy

    # Noise / randomness
    noise: float = 0.1
    seed: int = 1234
    use_random_seeds: bool = False
    #7881, 4523 6014 8257 4242 4042 5 7811 7830 7844
    random_seeds: List[int] = field(default_factory=lambda: [4, 5, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]) #
    init_bo_seed: int = 42
    n_kendall: int = 1000

    # Likelihood/modeling
    logit: bool = True
    multi: bool = False
    type_likelihood: bool = False
    
    plotting: bool = False
    save_results: bool = True
    output_path: str = 'outputs'
    save_every_trial: int = 1
    plot_path: str = 'plt.png'

    # Reward metric scoring
    non_human_score: bool = True
    score_metric: str = "lpips-clip-ssim" #clip, aesthetic, hpsv2, picscore, imagereward, lpips-clip-ssim
    aes_model_path: str = ".models/checkpoints/sac+logos+ava1-l14-linearMSE.pth"
    hps_model_path: str = "./models/checkpoints/HPS_v2_compressed.pt"
    target_img_path: str = ""



    #ImageGen parameters
    image_models: List[str] = field(default_factory=lambda: ["sdxl","pixart","flux","sd3"])
    prompts: str = ""
    title: str = ""
    num_inference_steps: int = 4
    img_seed : int = 864
    t_edit: float = 0.0
    delta: float = 0.2
    extra_kwargs: Dict[str, Dict[str,int]] = field(default_factory=lambda: {}) #"delta":{"resnet":0.1}
    use_cross_attention_masks: bool = False
    cross_maps_path = "./outputs"
    res_blk: str = "-".join([f"block_{i}" for i in range(3,14)])
    feat_blocks: str = "first" #first, last, all
    feats: str = "full-res" #conv, full-res, full-blk
    before_softmax : bool = False
    edit_resnet_uncond: bool = True
    attn_uncond: bool = False
    attn_blk: Dict[str, List[str]] = field(default_factory=lambda: {"cross":[], "self":[], "s-value":[f"block_{i}" for i in range(3,14)], "s-query":[f"block_{i}" for i in range(3,14)], "s-key":[f"block_{i}" for i in range(3,14)], "c-value":[], "c-query":[], "c-key":[]}) 
    attn_res: int = 32
    edit_type: str = "geometric" # composite, affine, geometric
    pad_crop: bool = True
    pad_kwargs: Dict[str, Any] = field(default_factory=lambda: {"mode":'constant',"value":0})
    visualize: bool = False
    print_blk: bool = False
    relative_factor: Dict[str, float] = field(default_factory=lambda: {"cross":1.0,"self":1.0,"resnet":1.0,"value":1.0, "query":1.0, "key":1.0})
    blending_alphas: Dict[str, Dict[str, Any]] = field(default_factory=lambda: (
        {
            "base_alphas": {
                "resnet": (0.5, 0.0), #mask - (0.9, 0.1) no-mask (0.4, 0.0)
                "self":   (0.8, 0.2),
                "value":  (0.8, 0.2),
                "query":  (0.8, 0.2),
                "key":  (0.8, 0.2),
                "cross":  (0.9, 0.1), #(0.6, 0.4)
            },
            "schedule": {"cross":"exp","self":"exp","resnet":"exp","value":"exp","query":"exp","key":"exp"},
            "weight_alphas_kwargs": {
                name: {
                    "lin": None,
                    "lin_w": {'t1':0.02, 't2': 0.12},
                    "exp": {'k': 5},
                    "cos": None,
                    "sig": {'k':5, 't_mid':0.07},
                    "piece": {'t1':0.02, 't2': 0.12},
                }
                for name in ["resnet", "self", "value", "query", "key","cross"]
            }
        }
    ))
    ratio: float = 0.9
    obj: str = f"objectives.low_dims.ImageGen.flux"

    ## Warping Affine parameters
    # geometric transform parameters
    blend: bool = False #blending warped and unwarped
    sr: int = 128
    _alpha: float = 2*torch.pi/3 #12   
    _s_alpha: float = 2*torch.pi/3
    _t: float = 0.75  #0.25   #0.15
    _s: float = 0.75 #0.5     #0.25
    _t_hom: float = 0.4
    _t_tps:float = 0.75   #0.4    #0.2
    _t_tps_for_afftps = None 
    tps_grid_size: int = 3
    tps_reg_factor:int = 0
    parametrize_with_gaussian: bool = False
    _horizontal_flip:bool = False
    transformation_types:List[str]= field(default_factory=lambda: ['affine', 'hom', 'tps', 'afftps']) #
    geometric_model: str = "afftps"
    # elastic deformation
    use_elastic: bool = True
    nbr_perturbations: int = 5
    sigma_mask: int = 7
    elastic_parameters: Dict[str, float] = field(default_factory=lambda: {"max_sigma": 0.04, "min_sigma": 0.1, "min_alpha": 1, "max_alpha": 0.4}) #{"max_sigma": 0.04, "min_sigma": 0.1, "min_alpha": 1, "max_alpha": 0.4}

    def __post_init__(self):
        """Validation and setup logic."""
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_initial_samples <= 0 or self.m <= 0:
            raise ValueError("n and m must be positive")
        if not self.acf_algos:
            raise ValueError("algos list cannot be empty")
        
        os.makedirs(self.output_path, exist_ok=True)

    def update(self, **kwargs):
        """Update config parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        self.__post_init__()


@dataclass
class MultiBOConfig_FLUX_promptset(MultiBOConfig_FLUX):
    """Additional promptset-specific parameters"""
    img_random_seeds: List[int] = field(default_factory=lambda: [7876, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536, 864]) #
    promptset_path: str = 'prompt_datasets/ae_prompts5/animals.txt'
    num_samples_per_prompt: int = 10
    prompts_per_category: int = -1
# Probit, Pairwise, EUBO, qEI, rand - [2, 7883, 4420, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, rand - [2, 7876, 4415, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, rand - [5, 7876, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, taf, rand - [5, 7881, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]
# Logit, Pairwise, EUBO, qEI, 2s-qEI, taf, 2s-taf, rand - [7847, 4, 4042, 6034, 8258, 1014, 9573, 226, 2683, 5536]