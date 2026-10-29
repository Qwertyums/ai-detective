"""Download the single experimental model into the workspace cache."""
from pathlib import Path
import os
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    folder = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=str(root / '.model-cache'),
                               allow_patterns=['*.json', 'merges.txt', 'vocab.json'], max_workers=2)
    # A bounded streaming timeout also works with proxies that stall on the
    # signed CDN URL returned by the hub's default large-file download.
    target = Path(folder) / 'model.safetensors'
    if not target.exists():
        import requests
        from huggingface_hub import HfApi
        import hashlib
        info = HfApi().model_info('Qwen/Qwen2.5-0.5B-Instruct', revision=Path(folder).name, files_metadata=True)
        metadata = next(s.lfs for s in info.siblings if s.rfilename == 'model.safetensors')
        expected, total = metadata.sha256, metadata.size
        url = f'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{Path(folder).name}/model.safetensors?download=true'
        temporary = target.with_suffix('.download')
        from concurrent.futures import ThreadPoolExecutor
        import time
        prefix = temporary.stat().st_size if temporary.exists() else 0
        if prefix > total:
            raise RuntimeError('Unexpected partial-file size')
        chunk_size = 32*1024*1024
        ranges = [(start, min(start+chunk_size, total)-1) for start in range(prefix,total,chunk_size)]
        def fetch(bounds):
            start, end = bounds
            part = target.with_name(f'weights-{start}-{end}.part')
            if part.exists() and part.stat().st_size == end-start+1:
                return part
            for attempt in range(3):
                try:
                    with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(20,30)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                            raise RuntimeError('CDN did not honor the requested byte range')
                        with part.open('wb') as stream:
                            for chunk in response.iter_content(1024*1024):
                                stream.write(chunk)
                    if part.stat().st_size != end-start+1:
                        raise RuntimeError('Incomplete range')
                    print(f'Completed bytes {start}-{end}', flush=True)
                    return part
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        print(f'Resuming from {prefix/1024**2:.0f} MiB with four bounded connections', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(fetch, ranges))
        assembled = target.with_suffix('.assembled')
        checksum = hashlib.sha256()
        with assembled.open('wb') as output:
            for source in ([temporary] if prefix else []) + parts:
                with source.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        checksum.update(chunk)
                        output.write(chunk)
        if checksum.hexdigest() != expected:
            raise RuntimeError('Weight checksum mismatch')
        assembled.replace(target)
        # These are verified task-owned temporary files inside the model cache.
        for source in ([temporary] if prefix else []) + parts:
            source.unlink()
    print(folder, flush=True)
