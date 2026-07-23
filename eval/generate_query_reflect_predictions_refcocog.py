import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometric-export-dir", type=str, required=True)
    parser.add_argument("--shared-mllm-path", type=str, default="")
    parser.add_argument("--ref-json-path", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--num-parts", type=int, default=-1)
    parser.add_argument("--resize-size", type=int, default=840)
    parser.add_argument("--sam-image-size", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=2007040)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=256)
    parser.add_argument("--reflect-max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--enable-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the reflection and accept/reject branch-selection stages (default: enabled).",
    )
    parser.add_argument(
        "--force-direct-after-reflection",
        action="store_true",
        default=False,
        help="Run reflection normally, but force every final prediction through the direct branch.",
    )
    parser.add_argument(
        "--enable-reflect-decision-logit-threshold",
        action="store_true",
        default=False,
        help=(
            "Select aligned/direct by the accept-vs-reject probability at the generated "
            "'Conclusion:' decision position. Disabled by default."
        ),
    )
    parser.add_argument(
        "--reflect-accept-probability-threshold",
        type=float,
        default=0.5,
        help="Choose aligned when the conclusion-position accept probability is at least this value.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--decision-mode", type=str, choices=["sign", "confidence_decision"], default="sign")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--save-branch-breakdown", action="store_true", default=True)
    parser.add_argument("--use_reject_direct_attention_mask", action="store_true", default=False)
    parser.add_argument("--use-direct-query-for-direct-branch", action="store_true", default=False)
    parser.add_argument("--direct-branch-prompt-mode", type=str, choices=["geometric", "query_reflect", "query_reflect_reason_only"], default="geometric")
    parser.add_argument("--save-reject-branch-visualizations", action="store_true", default=False)
    parser.add_argument("--reject-branch-visualization-limit", type=int, default=-1)
    parser.add_argument("--save-stage1-proposal-visualizations", action="store_true", default=False)
    parser.add_argument("--save-reject-direct-prompts", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    from training_scripts.eval.query_reflect_dataset import QueryReflectDataset
    from training_scripts.eval.query_reflect_eval_common import resolve_part_args, run_prediction_loop

    idx, num_parts = resolve_part_args(args.idx, args.num_parts)
    dataset = QueryReflectDataset(
        data_mode="refcocog_json",
        image_root=args.image_root,
        ref_json_path=args.ref_json_path,
        answer_resize=args.resize_size,
        sam_image_size=args.sam_image_size,
    )
    run_prediction_loop(
        dataset=dataset,
        output_dir=args.output_dir,
        idx=idx,
        num_parts=num_parts,
        geometric_export_dir=args.geometric_export_dir,
        shared_mllm_path=args.shared_mllm_path,
        resize_size=args.resize_size,
        sam_image_size=args.sam_image_size,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        stage1_max_new_tokens=args.stage1_max_new_tokens,
        reflect_max_new_tokens=args.reflect_max_new_tokens,
        enable_reflection=args.enable_reflection,
        force_direct_after_reflection=args.force_direct_after_reflection,
        enable_reflect_decision_logit_threshold=args.enable_reflect_decision_logit_threshold,
        reflect_accept_probability_threshold=args.reflect_accept_probability_threshold,
        confidence_threshold=args.confidence_threshold,
        decision_mode=args.decision_mode,
        limit=args.limit,
        save_branch_breakdown=args.save_branch_breakdown,
        use_reject_direct_attention_mask=args.use_reject_direct_attention_mask,
        save_reject_branch_visualizations=args.save_reject_branch_visualizations,
        save_stage1_proposal_visualizations=args.save_stage1_proposal_visualizations,
        save_reject_direct_prompts=args.save_reject_direct_prompts,
        use_direct_query_for_direct_branch=args.use_direct_query_for_direct_branch,
        direct_branch_prompt_mode=args.direct_branch_prompt_mode,
        reject_branch_visualization_limit=args.reject_branch_visualization_limit,
    )


if __name__ == "__main__":
    main()
