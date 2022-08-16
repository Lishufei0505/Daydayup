from .augmentations import letterbox
from .downloads import attempt_download
from .general import intersect_dicts, non_max_suppression, scale_coords, set_logging
from .torch_utils import select_device



__all__ = [
    "letterbox",
    "non_max_suppression",
    "intersect_dicts",
    "scale_coords",
    "set_logging",
    "attempt_download",
    "select_device",
  
]

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""


def notebook_init(verbose=True):
    # Check system software and hardware
    print('Checking setup...')

    import os
    import shutil

    from .general import check_requirements, emojis, is_colab
    from .torch_utils import select_device  # imports

    check_requirements(('psutil', 'IPython'))
    import psutil
    from IPython import display  # to display images and clear console output

    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)  # remove colab /sample_data directory

    # System info
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)'
    else:
        s = ''

    select_device(newline=False)
    print(emojis(f'Setup complete ✅ {s}'))
    return display
