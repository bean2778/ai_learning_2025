import numpy as np

def print_pattern(image, name=""):
    """Quick visualization of image pattern"""
    print(f"\n{name}:")
    for row in image:
        print(''.join('R' if px[0] > 200 else 'B' if px[2] > 200 else '.' for px in row))

def rotate_90(image: np.ndarray) -> np.ndarray:
    mid = np.transpose(image, (1, 0, 2))
    return np.flip(mid, axis=0)

def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=1)

def flip_vertical(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=0)

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    brighened = image * factor
    return np.clip(brighened, 0, 255).astype(np.uint8)

def crop_center(image: np.ndarray, crop_height: int, crop_width: int) -> np.ndarray:
    crop_height = min(image.shape[0], crop_height)
    crop_width = min(image.shape[1], crop_width)
    r1 = (image.shape[0] - crop_height) // 2
    c1 = (image.shape[1] - crop_height) // 2
    cropped = image[r1:r1 + crop_height, c1:c1 + crop_width, :]
    return cropped

def add_border(image: np.ndarray, border_width: int, border_color: int = 0) -> np.ndarray:
    return np.pad(image,
                  pad_width=((border_width, border_width),  # height
                             (border_width, border_width),  # width
                             (0, 0)),                       # channels
                  mode='constant',
                  constant_values=border_color)

def main():
    test_image = np.zeros((10, 10, 3), dtype=np.uint8)
    test_image[:, :5, 0] = 255  # Left half red
    test_image[:, 5:, 2] = 255  # Right half blue
    
    print("BEFORE:")
    print_pattern(test_image, "")
    
    print("\nSTEP 1 - After transpose:")
    mid = np.transpose(test_image, (1, 0, 2))
    print_pattern(mid, "")
    
    print("\nSTEP 2 - After flip axis=0:")
    result = np.flip(mid, axis=0)
    print_pattern(result, "")
    
    print("\nOR - After flip axis=1:")
    result2 = np.flip(mid, axis=1)
    print_pattern(result2, "")
        
if __name__ == "__main__":
    main()