import jsonargparse
import random
import numpy as np
import torch


def make():
    parser = jsonargparse.ArgumentParser()

    # General arguments
    parser.add_argument("--version", type=str, default="", help="experiment version")
    parser.add_argument(
        "--save_dir", default=None, type=str, help="Local output file name or path."
    )
    parser.add_argument(
        "--seed", default=123, type=int, help="Set the seed for the model and training."
    )
    parser.add_argument(
        "--device", default=0, type=int, help="cuda device ordinal to train on."
    )
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    parser.add_argument("--eval_mode", type=int, default=0, help="evaluation mode")
    parser.add_argument("--debug", type=int, default=0, help="debug mode")

    # Text arguments
    parser.add_argument(
        "--manual",
        type=str,
        choices=["none", "standard", "standardv2", "emma", "direct", "oracle"],
        help="which type of manuals to pass to the model",
    )
    parser.add_argument(
        "--gpt_groundings_path",
        default="chatgpt_groundings/chatgpt_grounding_few_shot.json",
        type=str,
        help="path to chatgpt groundings",
    )

    # World model arguments
    parser.add_argument(
        "--wm_weights_path",
        type=str,
        default=None,
        help="Path to world model state dict.",
    )
    parser.add_argument(
        "--hidden_size", default=256, type=int, help="World model hidden size."
    )
    parser.add_argument(
        "--encoder_num_heads", type=int, default=4, help="attribute embedding size"
    )
    parser.add_argument(
        "--encoder_layers", type=int, default=4, help="attribute embedding size"
    )
    parser.add_argument(
        "--encoder_tokens_per_block", type=int, default=3, help="tokens per blocks"
    )
    parser.add_argument("--encoder_max_blocks", type=int, default=3, help="max blocks")
    parser.add_argument(
        "--decoder_num_heads", type=int, default=4, help="attribute embedding size"
    )
    parser.add_argument(
        "--decoder_layers", type=int, default=4, help="action embedding size"
    )
    parser.add_argument(
        "--decoder_tokens_per_block", type=int, default=15, help="tokens per blocks"
    )
    parser.add_argument("--decoder_max_blocks", type=int, default=33, help="max blocks")
    parser.add_argument(
        "--special_init",
        type=int,
        default=1,
        help="customized parameter initialization",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        default="custom_dataset/wm_data_mixed_100k_train.pickle",
        help="path to the dataset file",
    )

    # Training arguments
    parser.add_argument(
        "--max_rollout_length",
        default=32,
        type=int,
        help="Max length of a rollout to train for",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch_size of training input"
    )
    parser.add_argument(
        "--max_batches", default=100000, type=int, help="max training batches"
    )
    parser.add_argument(
        "--grad_acc_steps",
        default=1,
        type=int,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm", default=10.0, type=float, help="max gradient norm"
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="World model learning rate"
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument(
        "--save_every_batches",
        type=int,
        default=None,
        help="number of batches between model saves",
    )
    parser.add_argument(
        "--shuffle_manual", type=int, default=1, help="Shuffle descriptions in a manual"
    )

    # Logging arguments
    parser.add_argument(
        "--log_every_batches",
        default=500,
        type=int,
        help="number of batches between evaluations",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="mode to run wandb in",
    )
    parser.add_argument("--use_wandb", type=int, default=0, help="log to wandb?")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity")

    # Environment arguments
    parser.add_argument(
        "--env.name", type=str, default="custom", help="Name of Gym environment"
    )
    parser.add_argument(
        "--env.stage", type=int, default=2, help="Stage of Messenger environment"
    )

    # Data generation arguments
    parser.add_argument(
        "--data_gen.save_path", type=str, default=None, help="path for saving data"
    )
    parser.add_argument(
        "--splits_path",
        default="custom_dataset/data_splits_final_with_messenger_names.json",
        type=str,
    )
    parser.add_argument(
        "--texts_path",
        default="../messenger/envs/texts/custom_text_splits/custom_text_splits_with_messenger_names.json",
        type=str,
    )
    parser.add_argument("--data_gen.num_train", default=100000, type=int)
    parser.add_argument("--data_gen.num_eval", default=500, type=int)
    parser.add_argument(
        "--data_gen.behavior_policy",
        default="mixed",
        type=str,
        help="behavior policy for generating rollouts",
    )
    parser.add_argument(
        "--data_gen.behavior_policy_weights_path",
        type=str,
        default=None,
        help="Path to policy weights",
    )

    # EMMA policy arguments
    parser.add_argument(
        "--emma_policy.hist_len",
        type=int,
        default=3,
        help="length of history used by EMMA policy",
    )
    parser.add_argument(
        "--emma_policy.base_arch",
        type=str,
        default="conv",
        choices=["conv", "transformer"],
        help="EMMA policy base architecture",
    )
    parser.add_argument(
        "--emma_policy.weights_path",
        type=str,
        default=None,
        help="Path to policy weights",
    )

    # downstream arguments
    parser.add_argument(
        "--downstream.splits_path",
        type=str,
        default="custom_dataset/data_splits_downstream.json",
        help="Path to downstream split file containing evaluation games",
    )

    parser.add_argument(
        "--downstream.fix_split",
        type=str,
        default=None,
        help="split for training on downstream task",
    )
    parser.add_argument(
        "--downstream.fix_game",
        type=int,
        default=None,
        help="game id for training on downstream task",
    )

    parser.add_argument(
        "--downstream.oracle_weights_path",
        type=str,
        default=None,
        help="Path to oracle policy weights",
    )

    parser.add_argument(
        "--downstream.task",
        type=str,
        default=None,
        choices=["imitation", "filtered_bc", "self_imitation"],
        help="Downstream task",
    )

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device}")

    # seed everything
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    return args
