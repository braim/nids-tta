# Moving from Colab to DGX Spark — Setup Notes

Notes from migrating a notebook (`NIDS_CTTA_VIS_8_1_1.ipynb`) from Google Colab to an
NVIDIA DGX Spark, accessed remotely via SSH from a Mac. Written so someone else can
repeat the same steps on their own Mac.

## What DGX Spark is, briefly

DGX Spark runs on NVIDIA's GB10 Grace Blackwell superchip: an **ARM64/aarch64** CPU
paired with a GPU, sharing **128GB of unified memory** (not separate CPU RAM + GPU VRAM
like a typical machine). Two things that trip people up coming from Colab:

- It's **ARM64, not x86_64**. Anything installed via a pinned platform-specific wheel
  from Colab (which runs on x86_64) will fail to install here.
- `nvidia-smi` reports `Memory-Usage: Not Supported` because of the unified memory
  architecture — use `free -h` or the DGX Dashboard to see memory usage instead.
- `MIG Mode: N/A` — no hardware-level isolation between users. If the box is shared,
  everyone is directly sharing the same GPU compute/memory pool.

## 1. Set up SSH access from your Mac

Add an alias to `~/.ssh/config` on your Mac so you don't retype connection details:

```
Host dgx-spark
    HostName <server-ip-or-hostname>
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

Then connect with just:

```
ssh dgx-spark
```

If you'll connect from multiple networks (home, office, cafe), consider Tailscale to
avoid dealing with changing IPs/firewalls.

## 2. Always run long jobs inside tmux

This is the biggest habit change from Colab: nothing dies just because your laptop
sleeps or your SSH session drops, but only if the job is running inside a persistent
terminal multiplexer.

```
tmux new -s myrun
# run your commands
```

Detach with `Ctrl+B` then `D`. Reattach later with `tmux attach -t myrun`.

## 3. Get the code onto the Spark

If the project is a git repo, just `git clone` it directly on the Spark once SSH'd in.
Otherwise, copy files over from the Mac with `scp` or `rsync`, e.g.:

```
scp ~/Downloads/your_notebook.ipynb dgx-spark:~/projects/
```

## 4. Create a Python virtual environment

Don't install into the system/shared Python — isolate per project:

```
python3 -m venv ~/projects/myenv
source ~/projects/myenv/bin/activate
```

## 5. Clean up Colab's `requirements.txt` before installing

Exporting `!pip freeze > requirements.txt` from Colab and installing it directly on
the Spark will fail repeatedly, because the freeze contains packages/wheels specific
to Colab's environment (x86_64, Colab-only internals) that don't exist or don't apply
here. Rather than fixing one error at a time, it's much faster to check what your
notebook/script *actually imports* and trim the file down to just that.

Categories of things we removed from the frozen `requirements.txt`, and why:

| Removed | Why |
|---|---|
| `google-colab`, anything under `/colabtools/` | Colab-internal packages, not on PyPI, not needed outside Colab |
| `jupyter_kernel_gateway` (googlecolab fork) | Colab-internal plumbing |
| `torch`, `torchvision`, `torchaudio`, `torchcodec` (pinned to `+cpu` x86_64 wheel URLs) | Built for x86_64 CPU-only — DGX Spark is ARM64 with a GPU. Installed separately instead (see step 6). |
| `torchao`, `torchdata`, `torchsummary`, `torchtune` (exact version pins) | Pinned to versions matched to the old x86_64 torch build; only reinstall these later, unpinned, if your code actually imports them |
| `cyipopt`, `pyomo` | Require a system-level Ipopt optimization library via `pkg-config` — not present on the Spark, and unused by this notebook |
| `GDAL` | Requires system GDAL library + `gdal-config` binary to compile — unused by this notebook |
| `polars-runtime-32` | Architecture-specific runtime pinned by the freeze; `polars` pulls the correct one for aarch64 on its own |

**How we identified what to keep:** extracted every top-level `import` statement from
the notebook, mapped import names to their PyPI package names (e.g. `sklearn` →
`scikit-learn`), and cross-referenced against the frozen `requirements.txt` to keep
only matching lines with their pinned versions. Final trimmed file for this notebook:

```
efficient-kan @ git+https://github.com/Blealtan/efficient-kan.git@<commit>
kagglehub==1.0.2
matplotlib==3.10.0
numpy==2.0.2
pandas==2.2.2
plotly==5.24.1
polars==1.35.2
scikit-learn==1.6.1
```

(torch deliberately excluded — see next step)

General rule of thumb: if pip errors on a package, check whether your code actually
imports it before spending time fixing the install. A lot of what `pip freeze` in
Colab captures is environment cruft, not real dependencies.

## 6. Install PyTorch separately, for aarch64 + CUDA 13

Don't rely on a frozen Colab wheel for torch — install fresh from PyTorch's official
CUDA 13 index, which has proper aarch64 builds for the GB10 chip:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

You may see a warning that `sm_121` (GB10's compute capability) isn't officially
listed as supported — expected on this newer hardware, safe to ignore.

## 7. Install the trimmed requirements

```
pip install -r requirements.txt
```

## 8. Verify the GPU actually works

```
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

Expect `True` and a device name like `NVIDIA GB10`.

## 9. Set up VS Code Remote-SSH (for an interactive, Colab-like notebook experience)

1. Install the **Remote - SSH** extension in VS Code on your Mac (also **Python** and
   **Jupyter** extensions if not already installed).
2. `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → select your SSH alias (e.g.
   `dgx-spark`). Accept the prompt to install Python/Jupyter extensions on the
   *remote* side too.
3. File → Open Folder → open the project folder on the Spark.
4. Install the kernel bridge inside your venv:
   ```
   source ~/projects/myenv/bin/activate
   pip install ipykernel
   python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
   ```
5. Open the `.ipynb` file. Click "Select Kernel" (top right) → "Select Another
   Kernel..." → **"Python Environments..."** → choose the entry matching your venv's
   path (confirm the exact path first with `which python` in an activated-venv
   terminal).

   Gotcha: if you instead browse to the interpreter path via a "Find/Browse" dialog,
   VS Code may launch its **"Create Environment"** wizard instead of just selecting
   the existing one — this creates a brand new, separate `.venv` (commonly dropped in
   the workspace root) rather than using the one you already set up. If that happens
   by accident:
   ```
   rm -rf <path-to-accidental>/.venv
   ```
   and check `.vscode/settings.json` for a `python.defaultInterpreterPath` entry
   pointing at it (remove if present), then `Cmd+Shift+P` → "Developer: Reload
   Window" before reselecting the correct kernel.

6. Run a cell (e.g. the torch check from step 8) to confirm the kernel is really
   using the right environment.

## 10. Clean up leftover Colab-specific cells

Any `!pip install ...` cells in the notebook that were meant to patch Colab's base
environment are now redundant if those packages are already in `requirements.txt`.
Comment them out (rather than delete) so the notebook still documents its
dependencies if it's ever run elsewhere:

```python
# !pip install -q git+https://github.com/Blealtan/efficient-kan.git
# !pip install -q polars kagglehub
```

## 11. Monitor the GPU while training runs

From a second SSH window:

```
watch -n 1 nvidia-smi
```

What to look for in the output:

- **Processes section** should show your venv's Python binary with non-zero GPU
  memory — confirms the job is actually using the GPU under the right environment.
- **GPU-Util / Power draw** should climb noticeably once training is in full swing
  (steady low numbers like ~8% util / ~13W for an extended period during what should
  be heavy compute suggests the GPU is starved, e.g. by a CPU-bound data loader
  rather than being the bottleneck itself).
- If this is a shared machine, always check `nvidia-smi` before launching a big job —
  there's no MIG isolation, so you and another user would be directly competing for
  the same GPU.

## Quick troubleshooting reference

| Error | Cause | Fix |
|---|---|---|
| `No such file or directory: '.../google_colab-1.0.0.tar.gz'` | Colab-internal package, path doesn't exist outside Colab | Remove the line from requirements.txt |
| `torch-....+cpu-...-x86_64.whl is not a supported wheel on this platform` | x86_64 CPU wheel on ARM64 hardware | Remove pinned torch lines; install from `--index-url https://download.pytorch.org/whl/cu130` |
| `pkg-config ... ipopt ... not found` | `cyipopt`/`pyomo` need system Ipopt library | Remove if unused by your code, or `apt install coinor-libipopt-dev` if genuinely needed |
| `Could not find gdal-config` | `GDAL` needs system GDAL library + dev headers | Remove if unused by your code |
