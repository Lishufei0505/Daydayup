import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path().resolve()
print(FILE)
ROOT = FILE
# ROOT = FILE.parents[0]  # YOLOrt root directory
print(print("----------------------------------------------------------"))
print(str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


from yolort.v5.utils import export, general, torch_utils, val, notebook_init

# from yolort.v5.utils import val
# from yolort.v5.utils import notebook_init

from yolort.v5.utils.general import LOGGER, check_yaml, file_size, print_args
from yolort.v5.utils.torch_utils import select_device


def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=640,  # inference size (pixels)
        batch_size=1,  # batch size
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        test=False,  # test exports only
        pt_only=True,  # test PyTorch only
        hard_fail=False,  # throw error on benchmark failure
):
    y, t = [], time.time()
    device = select_device(device)
    print (device)

    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, CPU, GPU)
        try:

            assert i not in (9, 10), 'inference not supported'  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == 'Darwin', 'inference only supported on macOS>=10.13'  # CoreML
            if 'cpu' in device.type:
                assert cpu, 'inference not supported on CPU'
            if 'cuda' in device.type:
                print("2222222222222222222222222222222222222")
                assert gpu, 'inference not supported on GPU'


            # Export
            if f == '-':
                print("3333333333333333")
                w = weights  # PyTorch format
            else:
                print("44444444444444444444444444444444444444444444")
                w = export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]  # all others
            assert suffix in str(w), 'export failed'


            # Validate
            print("666666666666666666666666")
            result = val.run(data, w, batch_size, imgsz, plots=False, device=device, task='benchmark', half=half)
            print("555555555555555555555555555555555555")
            metrics = result[0]  # metrics (mp, mr, map50, map, *losses(box, obj, cls))
            print('matrics:', metrics)
            speeds = result[2]  # times (preprocess, inference, postprocess)
            y.append([name, round(file_size(w), 1), round(metrics[3], 4), round(speeds[1], 2)])  # MB, mAP, t_inference
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f'Benchmark --hard-fail for {name}: {e}'
            LOGGER.warning(f'WARNING: Benchmark failure for {name}: {e}')
            y.append([name, None, None, None])  # mAP, t_inference
        if pt_only and i == 0:
            break  # break after PyTorch

    # Print results
    LOGGER.info('\n')
    parse_opt()
    print("111111111111111111111111111111111111")
    notebook_init()  # print system info
    c = ['Format', 'Size (MB)', 'mAP@0.5:0.95', 'Inference time (ms)'] if map else ['Format', 'Export', '', '']
    py = pd.DataFrame(y, columns=c)
    print(py)
    LOGGER.info(f'\nBenchmarks complete ({time.time() - t:.2f}s)')
    LOGGER.info(str(py if map else py.iloc[:, :2]))
    print("-----------------run Runing-------------------")
    return py


def test(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=640,  # inference size (pixels)
        batch_size=1,  # batch size
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        test=False,  # test exports only
        pt_only=False,  # test PyTorch only
        hard_fail=False,  # throw error on benchmark failure
):
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, gpu-capable)
        try:
            w = weights if f == '-' else \
                export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]  # weights
            assert suffix in str(w), 'export failed'
            y.append([name, True])
        except Exception:
            y.append([name, False])  # mAP, t_inference

    # Print results
    LOGGER.info('\n')
    parse_opt()
    notebook_init()  # print system info
    py = pd.DataFrame(y, columns=['Format', 'Export'])
    LOGGER.info(f'\nExports complete ({time.time() - t:.2f}s)')
    LOGGER.info(str(py))
    print ("-----------------test Runing-------------------")
    return py


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--test', action='store_true', help='test exports only')
    parser.add_argument('--pt-only', action='store_true', help='test PyTorch only')
    parser.add_argument('--hard-fail', action='store_true', help='throw error on benchmark failure')
    opt = parser.parse_args(args=[])
    opt.data = check_yaml(opt.data)  # check YAML
   # print (vars(opt))
    name = 'benchmarks'
    # print(name)
    # print(vars(opt))
    print_args(name, vars(opt))
    return opt


def main(opt):
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    print(ROOT)
    opt = parse_opt()
    print(opt.data)
    print ("opt :", opt)
    main(opt)