import sys
import subprocess
import torch
import re

from pathlib import Path
from typing import IO, Callable, List, Union

TQDM_PROGRESS_RE = re.compile(br"^ *([\d\.]+)%")

def readline(io: IO[bytes], *, until=b"\r") -> bytes:
  result = b""
  while True:
    b = io.read(1)
    if not b:
      return result
    result += b
    if b == until:
      return result

def demix(
    paths: List[Path],
    demix_dir: Path,
    device: Union[str, torch.device],
    progress_callback: Callable[[float], None] | None = None):
  """Demixes the audio file into its sources."""
  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'htdemucs' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    p = subprocess.Popen(
      [
        sys.executable, '-m', 'demucs.separate',
        '--out', demix_dir.as_posix(),
        '--name', 'htdemucs',
        '--device', str(device),
        *[path.as_posix() for path in todos],
      ],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.PIPE,
    )
    assert p.stderr is not None
    progress = 0.0
    processed_count = 0
    while p.poll() is None:
      line = readline(p.stderr)
      match = TQDM_PROGRESS_RE.match(line)
      if match:
        new_progress = float(match.group(1)) / 100
        if new_progress < 0.001 and progress > 0:
          processed_count += 1
        progress = new_progress
      try:
        if progress_callback:
          progress_callback((processed_count + progress) / len(todos))
      except Exception:
        p.terminate()
        raise
    if p.returncode != 0:
      raise subprocess.CalledProcessError(p.returncode, p.args)

  return demix_paths
