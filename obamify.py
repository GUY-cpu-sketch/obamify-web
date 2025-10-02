import cv2
import numpy as np

def obamify(source_img, target_img, alpha=0.5):
    """
    Morphs the source image toward the target image using simple color blending.

    Args:
        source_img (np.ndarray): source image (BGR)
        target_img (np.ndarray): target image (BGR)
        alpha (float): blending factor (0.0 = source, 1.0 = target)

    Returns:
        np.ndarray: blended image
    """
    # Resize target to match source
    target_resized = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))
    
    # Convert to float for blending
    src_f = source_img.astype(np.float32)
    tgt_f = target_resized.astype(np.float32)
    
    # Blend the images
    blended = cv2.addWeighted(src_f, 1 - alpha, tgt_f, alpha, 0)
    
    # Convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended
