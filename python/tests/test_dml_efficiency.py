# from loguru import logger
# import sys
from pathlib import Path
import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_onnxruntime import RapidOCR

from pathlib import Path

import cv2
import numpy as np
import timeit
import time
import pytest
from loguru import logger
# from base_module import BaseModule

engine = RapidOCR(None, cls_use_dml=True, det_use_dml=True, rec_use_dml=True)
tests_dir = root_dir / "tests" / "test_bigimg"
# img_path = tests_dir / "ch_en_num.jpg"
package_name = "rapidocr_onnxruntime"

def benchmark():
    global engine
    img_list = tests_dir.glob("*.png")
    if not engine:
        raise Exception("Engine not initialized")
    _ = [engine(img_path) for img_path in img_list]


@pytest.mark.my_custom_tests
def test_all_image():
    """
    Test all images in the test directory
    Used for bechmarking batch inference
    Since standard OCR ONNX model not support batch inference, we add a loop to simulate batch inference
    """
    dml_time = timeit.timeit(
        benchmark, number=1, globals=globals())
    logger.info (f"DML time: {dml_time:.2f}s")

    # del engine
    
    # engine = RapidOCR(None, cls_use_dml=False, det_use_dml=False, rec_use_dml=False)
    global engine
    engine = RapidOCR(None, cls_use_dml=False, det_use_dml=False, rec_use_dml=False)
    ort_time = timeit.timeit(
        benchmark, number=1, globals=globals())
    
    # print(f"ORT time: {ort_time:.2f}s")
    logger.info (f"ORT time: {ort_time:.2f}s")
    assert ort_time > dml_time
