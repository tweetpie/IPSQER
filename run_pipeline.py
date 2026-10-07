#!/usr/bin/env python3
"""
Run the IPSQER pipeline end to end.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    name: str
    display_name: str
    script: Path
    expected_outputs: tuple[Path, ...]
    note: str = ""


STAGES: tuple[Stage, ...] = (
    Stage(
        name="multiintent_generation",
        display_name="Multi-Intent Generation",
        script=REPO_ROOT / "src" / "multiintent_generation.py",
        expected_outputs=(
            REPO_ROOT / "output" / "multiintent_generation" / "query_subtopics_list.json",
        ),
        note="Uses the configured LLM endpoint.",
    ),
    Stage(
        name="grounded_intent_validation_1",
        display_name="BM25 Grounding Retrieval",
        script=REPO_ROOT / "src" / "grounded_intent_validation_1.py",
        expected_outputs=(REPO_ROOT / "output" / "grounded_intent_validation_1" / "retrieval_grounding",),
        note="Requires the BM25 Wikipedia index.",
    ),
    Stage(
        name="grounded_intent_validation_2",
        display_name="Grounding Scores",
        script=REPO_ROOT / "src" / "grounded_intent_validation_2.py",
        expected_outputs=(
            REPO_ROOT / "output" / "grounded_intent_validation_2" / "grounding_scores_summary.csv",
            REPO_ROOT / "output" / "grounded_intent_validation_2" / "grounding_scores_filtered.csv",
        ),
        note="Loads a cross-encoder model.",
    ),
    Stage(
        name="intent_specific_query_expansion",
        display_name="Intent-Specific Query Expansion",
        script=REPO_ROOT / "src" / "intent_specific_query_expansion.py",
        expected_outputs=(REPO_ROOT / "output" / "intent_specific_query_expansion" / "intent_specific_expansions.json",),
        note="Uses the configured LLM endpoint.",
    ),
    Stage(
        name="per_intent_retrieval_bm25",
        display_name="BM25 Retrieval",
        script=REPO_ROOT / "src" / "per-intent_retrieval_1.py",
        expected_outputs=(REPO_ROOT / "output" / "per_intent_retrieval_1" / "bm25_results.json",),
        note="Requires the BM25 Wikipedia index.",
    ),
    Stage(
        name="per_intent_retrieval_dense",
        display_name="Dense Retrieval",
        script=REPO_ROOT / "src" / "per-intent_retrieval_2.py",
        expected_outputs=(REPO_ROOT / "output" / "per_intent_retrieval_2" / "deep_results.json",),
        note="Requires wiki_output and wiki_faiss_gpu.",
    ),
    Stage(
        name="combine_retrieval_results",
        display_name="Combining Retrieval List",
        script=REPO_ROOT / "src" / "per-intent_retrieval_3.py",
        expected_outputs=(REPO_ROOT / "output" / "per_intent_retrieval_3" / "combined_results.json",),
    ),
    Stage(
        name="reranking",
        display_name="Reranking",
        script=REPO_ROOT / "src" / "reranking.py",
        expected_outputs=(REPO_ROOT / "output" / "reranking" / "final_reranked_results.json",),
        note="Loads a cross-encoder model.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all IPSQER pipeline scripts in dependency order."
    )
    parser.add_argument(
        "--from-stage",
        choices=[stage.name for stage in STAGES],
        default=STAGES[0].name,
        help="First stage to run.",
    )
    parser.add_argument(
        "--to-stage",
        choices=[stage.name for stage in STAGES],
        default=STAGES[-1].name,
        help="Last stage to run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a stage when all of its expected outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stages that would run without executing them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for each stage.",
    )
    parser.add_argument(
        "--time-file",
        default=str(REPO_ROOT / "output" / "pipeline_times.csv"),
        help="CSV file where per-stage timing results will be saved.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete each selected stage's corresponding output contents and exit.",
    )
    parser.add_argument(
        "--clean-before-run",
        action="store_true",
        help="Delete each selected stage's corresponding output contents before running it.",
    )
    parser.add_argument(
        "--query",
        help="Run the pipeline for one query instead of reading queries/combined_queries.csv.",
    )
    parser.add_argument(
        "--query-number",
        default="single-1",
        help="Query id to use with --query.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run each selected stage through Scalene and save one profile per stage.",
    )
    parser.add_argument(
        "--profile-dir",
        default=str(REPO_ROOT / "output" / "profiles"),
        help="Directory where Scalene profile JSON files will be saved.",
    )
    parser.add_argument(
        "--scalene",
        default=None,
        help="Scalene executable to use when --profile is set. By default, uses '--python -m scalene'.",
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model for both multi-intent generation and intent-specific expansion.",
    )
    parser.add_argument(
        "--multiintent-model",
        help="LLM model for multi-intent generation only.",
    )
    parser.add_argument(
        "--expansion-model",
        help="LLM model for intent-specific query expansion only.",
    )
    parser.add_argument(
        "--max-generated-intents",
        type=int,
        help="Maximum number of intents to generate per query.",
    )
    parser.add_argument(
        "--top-pair-fraction",
        type=float,
        help="Top pair fraction threshold used by grounded_intent_validation_2.py.",
    )
    parser.add_argument(
        "--lambda-param",
        type=float,
        help="Reranking relevance/diversity trade-off used by reranking.py.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        help="Hugging Face cross-encoder model for both grounding scores and reranking.",
    )
    parser.add_argument(
        "--grounding-model",
        help="Hugging Face cross-encoder model for grounded_intent_validation_2.py only.",
    )
    parser.add_argument(
        "--reranker-model",
        help="Hugging Face cross-encoder model for reranking.py only.",
    )
    return parser.parse_args()


def selected_stages(from_stage: str, to_stage: str) -> tuple[Stage, ...]:
    names = [stage.name for stage in STAGES]
    start = names.index(from_stage)
    end = names.index(to_stage)

    if start > end:
        raise SystemExit("--from-stage must come before or equal --to-stage.")

    return STAGES[start : end + 1]


def outputs_exist(stage: Stage) -> bool:
    return bool(stage.expected_outputs) and all(path.exists() for path in stage.expected_outputs)


def stage_output_paths(stage: Stage) -> tuple[Path, ...]:
    paths: list[Path] = []

    for output_path in stage.expected_outputs:
        if output_path.exists() and output_path.is_dir():
            candidate = output_path
        else:
            candidate = output_path.parent

        if candidate not in paths:
            paths.append(candidate)

    return tuple(paths)


def clean_stage_output(stage: Stage, dry_run: bool = False) -> None:
    for output_path in stage_output_paths(stage):
        resolved_path = output_path.resolve()

        try:
            resolved_path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing to clean output outside project folder: {resolved_path}"
            ) from exc

        if not output_path.exists():
            if dry_run:
                print(f"Would clean: {output_path} (does not exist yet)")
            continue

        if dry_run:
            print(f"Would clean contents of: {output_path}")
            continue

        if output_path.is_file():
            output_path.unlink()
            print(f"Deleted file: {output_path}")
            continue

        for child in output_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        print(f"Deleted contents of: {output_path}")


def print_header(stage_number: int, total: int, stage: Stage) -> None:
    print()
    print("=" * 80)
    print(f"[{stage_number}/{total}] {stage.display_name}")
    print(f"stage: {stage.name}")
    print(f"script: {stage.script}")
    if stage.note:
        print(f"note: {stage.note}")
    print("=" * 80)
    print(flush=True)


def run_stage(
    stage: Stage,
    python_executable: str,
    extra_env: dict[str, str] | None = None,
    profile: bool = False,
    profile_dir: Path | None = None,
    scalene_executable: str | None = None,
) -> Path | None:
    if not stage.script.exists():
        raise FileNotFoundError(f"Stage script not found: {stage.script}")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    if profile:
        if profile_dir is None:
            raise ValueError("profile_dir is required when profile=True.")

        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_file = profile_dir / f"{stage.name}.json"
        profile_html = profile_dir / f"{stage.name}.html"
        default_profile_json = REPO_ROOT / "profile.json"
        default_profile_html = REPO_ROOT / "profile.html"

        for default_profile in (default_profile_json, default_profile_html):
            if default_profile.exists():
                default_profile.unlink()

        if scalene_executable:
            command = [
                scalene_executable,
                "--cli",
                "--json",
                "--outfile",
                str(profile_file),
                str(stage.script),
            ]
        else:
            command = [
                python_executable,
                "-m",
                "scalene",
                "--cli",
                "--json",
                "--outfile",
                str(profile_file),
                str(stage.script),
            ]
        print(f"Profiling script: {stage.script}")
        print(f"Scalene report target: {profile_file}")
    else:
        command = [python_executable, str(stage.script)]
        profile_file = None

    subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )

    if profile and profile_file is not None:
        default_profile_json = REPO_ROOT / "profile.json"
        default_profile_html = REPO_ROOT / "profile.html"

        if not profile_file.exists() and default_profile_json.exists():
            shutil.move(str(default_profile_json), str(profile_file))

        if default_profile_html.exists():
            shutil.move(str(default_profile_html), str(profile_html))

    return profile_file


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def timing_row(
    stage: Stage,
    status: str,
    start_time: float,
    end_time: float,
) -> dict[str, object]:
    duration_seconds = end_time - start_time

    return {
        "stage": stage.display_name,
        "file": stage.script.name,
        "status": status,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
        "duration_seconds": round(duration_seconds, 3),
        "duration": format_duration(duration_seconds),
    }


def write_time_file(time_file: Path, rows: list[dict[str, object]]) -> None:
    time_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stage",
        "file",
        "status",
        "start_time",
        "end_time",
        "duration_seconds",
        "duration",
    ]

    with time_file.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_env_override(
    env: dict[str, str],
    name: str,
    value: object | None,
) -> None:
    if value is not None:
        env[name] = str(value)


def main() -> int:
    args = parse_args()
    stages = selected_stages(args.from_stage, args.to_stage)
    time_file = Path(args.time_file)
    profile_dir = Path(args.profile_dir)
    timing_rows: list[dict[str, object]] = []
    clean_before_run = args.clean_before_run or bool(args.query)
    extra_env: dict[str, str] = {}

    if args.query:
        extra_env["IPSQER_SINGLE_QUERY"] = args.query
        extra_env["IPSQER_SINGLE_QUERY_NUMBER"] = args.query_number

    add_env_override(extra_env, "IPSQER_MODEL", args.llm_model)
    add_env_override(extra_env, "IPSQER_MULTIINTENT_MODEL", args.multiintent_model)
    add_env_override(extra_env, "IPSQER_EXPANSION_MODEL", args.expansion_model)
    add_env_override(extra_env, "IPSQER_MAX_GENERATED_INTENTS", args.max_generated_intents)
    add_env_override(extra_env, "IPSQER_TOP_PAIR_FRACTION", args.top_pair_fraction)
    add_env_override(extra_env, "IPSQER_LAMBDA_PARAM", args.lambda_param)
    add_env_override(extra_env, "IPSQER_CROSS_ENCODER_MODEL_NAME", args.cross_encoder_model)
    add_env_override(extra_env, "IPSQER_GROUNDING_MODEL_NAME", args.grounding_model)
    add_env_override(extra_env, "IPSQER_RERANKER_MODEL_NAME", args.reranker_model)

    print("IPSQER pipeline runner")
    print(f"repo: {REPO_ROOT}")
    print(f"python: {args.python}")
    print()
    print("Important: paths are resolved relative to this project folder.")
    print("Make sure BM25 index, FAISS index, Wikipedia output, and API keys are available.")
    print(f"timing file: {time_file}")
    if args.profile:
        print("profiling: enabled")
        print(f"profile dir: {profile_dir}")
        if args.scalene:
            print(f"scalene: {args.scalene}")
        else:
            print(f"scalene: {args.python} -m scalene")
    if args.query:
        print(f"single query: {args.query}")
        print(f"single query number: {args.query_number}")
    override_keys = [
        key
        for key in sorted(extra_env)
        if key.startswith("IPSQER_")
        and key
        not in {
            "IPSQER_SINGLE_QUERY",
            "IPSQER_SINGLE_QUERY_NUMBER",
        }
    ]
    if override_keys:
        print("overrides:")
        for key in override_keys:
            print(f"  {key}={extra_env[key]}")
    if args.clean_output:
        print("clean output: enabled")
    if clean_before_run:
        print("clean before run: enabled")

    if args.clean_output:
        for index, stage in enumerate(stages, start=1):
            print_header(index, len(stages), stage)
            clean_stage_output(stage, dry_run=args.dry_run)

        print()
        print("Output cleanup complete.")
        return 0

    for index, stage in enumerate(stages, start=1):
        print_header(index, len(stages), stage)

        if clean_before_run:
            clean_stage_output(stage, dry_run=args.dry_run)

        if args.skip_existing and outputs_exist(stage):
            print(f"Skipping {stage.name}: expected output already exists.")
            now = time.time()
            timing_rows.append(timing_row(stage, "skipped", now, now))
            continue

        if args.dry_run:
            if args.profile:
                profile_file = profile_dir / f"{stage.name}.json"
                if args.scalene:
                    print(
                        "Would run: "
                        f"{args.scalene} --json --outfile {profile_file} {stage.script}"
                    )
                else:
                    print(
                        "Would run: "
                        f"{args.python} -m scalene --json --outfile {profile_file} {stage.script}"
                    )
            else:
                print(f"Would run: {args.python} {stage.script}")
            now = time.time()
            timing_rows.append(timing_row(stage, "dry_run", now, now))
            continue

        start_time = time.time()
        try:
            profile_file = run_stage(
                stage,
                args.python,
                extra_env=extra_env,
                profile=args.profile,
                profile_dir=profile_dir,
                scalene_executable=args.scalene,
            )
        except subprocess.CalledProcessError as exc:
            end_time = time.time()
            timing_rows.append(timing_row(stage, "failed", start_time, end_time))
            write_time_file(time_file, timing_rows)
            print()
            print(f"Pipeline stopped: {stage.name} failed with exit code {exc.returncode}.")
            print(f"Timing report saved to: {time_file}")
            return exc.returncode

        end_time = time.time()
        timing_rows.append(timing_row(stage, "success", start_time, end_time))
        if profile_file is not None:
            if profile_file.exists():
                print(f"Scalene report created: {profile_file}")
            else:
                print(f"Warning: Scalene report was not found after stage finished: {profile_file}")
        print(f"Finished {stage.display_name} in {format_duration(end_time - start_time)}.")

    write_time_file(time_file, timing_rows)

    print()
    print("Pipeline complete.")
    print(f"Timing report saved to: {time_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
