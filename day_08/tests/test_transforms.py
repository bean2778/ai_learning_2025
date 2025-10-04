import pytest
import numpy as np
from image_transforms import rotate_90, flip_horizontal, flip_vertical, adjust_brightness


def test_rotate_90_basic():
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :50, 0] = 255
    test_image[:, 50:, 2] = 255

    rotated = rotate_90(test_image)
    assert rotated.shape == (100, 100, 3)
    assert rotated[0, 0, 2] == 255

def test_flip_horizontal():
    test_image = np.zeros((10, 10), dtype=np.uint8)
    test_image[:5, :5] = 0
    test_image[:5, 5:] = 1
    test_image[5:, :5] = 2
    test_image[5:, 5:] = 3
    
    flipped = flip_horizontal(test_image)
    assert flipped[0, 0] == 1
    assert flipped[0, 9] == 0
    assert flipped[9, 0] == 3
    assert flipped[9, 9] == 2

def test_flip_vertical():
    test_image = np.zeros((10, 10), dtype=np.uint8)
    test_image[:5, :5] = 0
    test_image[:5, 5:] = 1
    test_image[5:, :5] = 2
    test_image[5:, 5:] = 3
    
    flipped = flip_vertical(test_image)
    assert flipped[0, 0] == 2
    assert flipped[0, 9] == 3
    assert flipped[9, 0] == 0
    assert flipped[9, 9] == 1

def test_adjust_brightness():
    test_image = np.zeros((10, 10), dtype=np.int32)
    test_image[:5, :5] = 0
    test_image[:5, 5:] = -1
    test_image[5:, :5] = 10
    test_image[5:, 5:] = 200

    adjusted = adjust_brightness(test_image, 2)
    assert adjusted[0, 0] == 0
    assert adjusted[0, 9] == 0
    assert adjusted[9, 0] == 20
    assert adjusted[9, 9] == 255