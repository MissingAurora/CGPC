"""Generate structured teacher CoT in mock or OpenAI-compatible mode.

Mock mode is only for testing the offline pipeline; it does not inspect images
and must not be treated as teacher-VLM data.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cot_common import (
    coarse_position,
    compact_json,
    parse_teacher_json,
    read_jsonl,
    safe_name,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=["mock", "openai-compatible"], default="mock")
    parser.add_argument(
        "--model",
        default=os.getenv("TEACHER_MODEL", "qwen3-vl-flash"),
        help="百炼模型ID，默认qwen3-vl-flash",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("TEACHER_API_URL") or os.getenv("DASHSCOPE_BASE_URL"),
        help="OpenAI兼容base_url；也可使用TEACHER_API_URL或DASHSCOPE_BASE_URL",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="密钥环境变量名；默认依次查找TEACHER_API_KEY、DASHSCOPE_API_KEY",
    )
    parser.add_argument("--prompt-file", type=Path, default=Path(__file__).resolve().parent / "prompts/teacher_system.txt")
    parser.add_argument("--prompt-version", default="grounded")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="教师模型单次最大输出token数，默认4096",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=2048,
        help="超大图片上传前缩放的最长边，默认2048；不修改原图",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="超大图片转JPEG时的质量，默认85",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def task_steps(sample: dict[str, Any]) -> list[str]:
    task = sample["task_type"]
    target = sample["target"]
    if task == "all_bales":
        return [
            "扫描完整画面并区分边界独立的棉包实例。",
            "逐个检查相邻实例的轮廓与接缝，避免把相邻棉包合并或遗漏。",
        ]
    if task == "selected_rows":
        return [
            "以题目指定的最近堆垛为参照理解行结构。",
            f"仅保留“{target}”对应的棉包，排除其他堆垛和其他行。",
        ]
    return [
        "以题目指定的最近堆垛为参照理解行列关系。",
        f"只保留“{target}”对应的唯一棉包。",
    ]


def mock_teacher(sample: dict[str, Any], prompt_version: str = "grounded") -> dict[str, Any]:
    answer = sample["ground_truth"]
    objects = []
    for index, (box, label) in enumerate(
        zip(answer["bbox_2d"], answer["labels"]), start=1
    ):
        obj = {"index": index, "bbox_2d": box, "label": label}
        if prompt_version != "grounded":
            obj.update(
                {
                    "position": "画面" + coarse_position(box),
                    "visual_evidence": (
                        "mock模式未读取图像，仅根据人工标注填充结构，视觉证据需要真实教师VLM生成。"
                    ),
                }
            )
        objects.append(obj)
    return {
        "task_understanding": f"定位{sample['target']}并区分front与side。",
        "scene_analysis": "mock模式不进行真实画面分析。",
        "selection_reasoning": task_steps(sample),
        "objects": objects,
        "verification": {
            "target_count": sample["target_count"],
            "front_count": sample["front_count"],
            "side_count": sample["side_count"],
        },
        "final_answer": answer,
    }


def image_data_url(path: Path, max_image_side: int, jpeg_quality: int) -> str:
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    # Base64 expands binary data by roughly one third. Re-encode only large
    # inputs so the OpenAI-compatible request remains below the gateway limit.
    if len(data) > 4 * 1024 * 1024:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError("处理超大图片需要Pillow：python -m pip install Pillow") from exc
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            if max(image.size) > max_image_side:
                image.thumbnail(
                    (max_image_side, max_image_side), Image.Resampling.LANCZOS
                )
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, "white")
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )
            data = buffer.getvalue()
            mime = "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def user_payload(sample: dict[str, Any], prompt_version: str) -> str:
    answer_key = (
        "answer_boxes_and_labels"
        if prompt_version == "grounded"
        else "ground_truth"
    )
    value = {
        "sample_id": sample["id"],
        "task_type": sample["task_type"],
        "target": sample["target"],
        "question": sample["question"],
        answer_key: sample["ground_truth"],
    }
    return "请依据图像和以下数据生成视觉推理：\n" + json.dumps(
        value, ensure_ascii=False, indent=2
    )


def call_openai_compatible(
    sample: dict[str, Any], args: argparse.Namespace, system_prompt: str
) -> tuple[str, dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("请先安装openai包：python -m pip install openai") from exc
    key_env_names = (
        [args.api_key_env]
        if args.api_key_env
        else ["TEACHER_API_KEY", "DASHSCOPE_API_KEY"]
    )
    api_key = next((os.getenv(name) for name in key_env_names if os.getenv(name)), None)
    if not api_key:
        raise RuntimeError("未找到TEACHER_API_KEY或DASHSCOPE_API_KEY环境变量")
    api_key = api_key.strip()
    if "你的" in api_key or api_key.lower() in {"your_api_key", "api_key", "xxx"}:
        raise RuntimeError(
            "DASHSCOPE_API_KEY仍是示例占位符，请替换为百炼控制台创建的真实API Key"
        )
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "API Key包含中文或其他非ASCII字符；请填写真实API Key，不要填写“你的API密钥”"
        ) from exc
    if not args.model:
        raise RuntimeError("未设置--model或TEACHER_MODEL")
    if not args.base_url:
        raise RuntimeError("未设置--base-url、TEACHER_API_URL或DASHSCOPE_BASE_URL")
    args.base_url = args.base_url.strip()
    parsed_url = urlparse(args.base_url)
    if "你的" in args.base_url or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise RuntimeError(
            "TEACHER_API_URL不是有效地址，请替换为此前成功调用API时使用的真实base_url"
        )
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    image_path = (args.root.resolve() / sample["image"]).resolve()
    response = client.chat.completions.create(
        model=args.model,
        temperature=0.1,
        max_tokens=args.max_output_tokens,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_payload(sample, args.prompt_version)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(
                                image_path, args.max_image_side, args.jpeg_quality
                            )
                        },
                    },
                ],
            },
        ],
    )
    text = response.choices[0].message.content
    usage = response.usage.model_dump() if response.usage else {}
    return text, usage


def process_one(
    sample: dict[str, Any], args: argparse.Namespace, system_prompt: str
) -> tuple[str, str]:
    output_path = args.output_dir / f"{safe_name(sample['id'])}.json"
    if args.resume and output_path.is_file():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                return sample["id"], "skipped"
        except json.JSONDecodeError:
            pass
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            if args.mode == "mock":
                teacher = mock_teacher(sample, args.prompt_version)
                response_text = compact_json(teacher)
                usage = {}
                model = "mock-no-image"
            else:
                response_text, usage = call_openai_compatible(sample, args, system_prompt)
                model = args.model
            # Do not mark truncated or otherwise malformed model output as success.
            parse_teacher_json(response_text)
            record = {
                "id": sample["id"],
                "status": "success",
                "mode": args.mode,
                "teacher_model": model,
                "prompt_version": args.prompt_version,
                "attempt": attempt,
                "response_text": response_text,
                "usage": usage,
            }
            write_json(output_path, record)
            return sample["id"], "success"
        except Exception as exc:  # keep individual failures from stopping the batch
            last_error = f"{type(exc).__name__}: {exc}"
    write_json(
        output_path,
        {
            "id": sample["id"],
            "status": "failed",
            "mode": args.mode,
            "teacher_model": args.model,
            "prompt_version": args.prompt_version,
            "attempt": args.max_retries,
            "error": last_error,
        },
    )
    return sample["id"], "failed"


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    if not args.prompt_file.is_absolute():
        args.prompt_file = args.root / args.prompt_file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = args.prompt_file.read_text(encoding="utf-8")
    samples = read_jsonl(args.input)
    if args.limit > 0:
        samples = samples[: args.limit]
    counts = {"success": 0, "failed": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process_one, sample, args, system_prompt) for sample in samples]
        for completed, future in enumerate(as_completed(futures), start=1):
            sample_id, status = future.result()
            counts[status] += 1
            print(f"[{completed}/{len(samples)}] {status}: {sample_id}", flush=True)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
