import pytest
import numpy as np
from image_transforms import (
    rotate_90, flip_horizontal, 
    flip_vertical, 
    adjust_brightness, 
    crop_center,
    add_border
)


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

def test_crop_center():
    # 10x10 image, each pixel = row*10 + col
    test_image = np.arange(100).reshape(10, 10, 1).astype(np.uint8)
    
    # Crop 4x4 from center
    cropped = crop_center(test_image, 4, 4)
    
    print("Cropped shape:", cropped.shape)  # Should be (4, 4, 1)
    print("Cropped values:\n", cropped[:, :, 0])
    
    # Center 4x4 should start at row 3, col 3
    # So top-left should be 33, top-right should be 36
    assert cropped[0, 0, 0] == 33
    assert cropped[0, 3, 0] == 36
    assert cropped[3, 0, 0] == 63

def test_add_border():
    # Create small test image (4x4x3)
    test_image = np.ones((4, 4, 3), dtype=np.uint8) * 100
    
    # Add 2-pixel black border
    bordered = add_border(test_image, 2, border_color=0)
    
    # Check shape increased correctly
    assert bordered.shape == (8, 8, 3)  # 4 + 2*2 = 8
    
    # Check border is black (0)
    assert bordered[0, 0, 0] == 0  # Top-left corner
    assert bordered[0, 4, 1] == 0  # Top edge
    assert bordered[7, 7, 2] == 0  # Bottom-right corner
    assert bordered[4, 0, 0] == 0  # Left edge
    assert bordered[4, 7, 1] == 0  # Right edge
    
    # Check center is preserved (original image at [2:6, 2:6])
    assert bordered[2, 2, 0] == 100  # Top-left of original
    assert bordered[5, 5, 2] == 100  # Bottom-right of original
    assert bordered[3, 4, 1] == 100  # Middle of original

def test_add_border_white():
    # Test with white border
    test_image = np.zeros((3, 3, 3), dtype=np.uint8)
    bordered = add_border(test_image, 1, border_color=255)
    
    assert bordered.shape == (5, 5, 3)
    assert bordered[0, 0, 0] == 255  # Border is white
    assert bordered[2, 2, 0] == 0    # Center is still black

def test_add_border_zero_width():
    # Edge case: no border
    test_image = np.ones((5, 5, 3), dtype=np.uint8)
    bordered = add_border(test_image, 0)
    
    assert bordered.shape == (5, 5, 3)
    assert np.array_equal(bordered, test_image)  # Should be unchanged