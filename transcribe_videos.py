import subprocess
from pathlib import Path

import click
from loguru import logger

import stable_whisper


@click.group
def cli():
    pass

@cli.command
@click.option("-i", "--input-dir")
@click.option("-o", "--output-dir", default=None)
@click.option("-m", "--model-type", default="large-v2")
@click.option("-l", "--language", default=None, type=str)
@click.option("-s", "--subtitle-suffix", default=".srt", type=str)
@click.option("-f", "--file-type", help="File types to filter", type=str, default="mp4")
def transcribe(input_dir, output_dir, model_type, language, subtitle_suffix, file_type):
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    files = list(input_dir.glob(f"*.{file_type}"))
    files.sort()
    logger.info(f"Transcribing {len(files)} with language '{language}'")
    logger.info(f"Loading model {model_type}")
    model = stable_whisper.load_model(model_type)
    for f in files:
        out_file = output_dir / f.with_suffix(subtitle_suffix).name
        logger.info(f"Transcribing episode to {out_file}")
        subs = model.transcribe(str(f), language=language)
        subs.to_srt_vtt(str(out_file))

@cli.command
@click.option("-i", "--input-dir")
@click.option("-o", "--output-dir")
@click.option("-f", "--file-type", help="File types to filter", type=str, default="mp4")
def convert(input_dir, output_dir, file_type):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    files = list(input_dir.rglob(f"*.{file_type}"))
    files.sort()
    cmd = 'ffmpeg -threads 0 -i "{input_file}" -c:a aac "{output_file}"'
    for f in files:
        cur_cmd = cmd.format(input_file=f, output_file=output_dir / f.with_suffix('.aac').name)
        logger.info(f"Running command {cur_cmd}")
        subprocess.check_call(cur_cmd, shell=True)


if __name__ == "__main__":
    cli()
