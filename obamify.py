import cv2
import numpy as np

def obamify(source_img, target_img, alpha=0.6):
    """
    Morphs source image toward target image.
    alpha = 0.0 -> only source
    alpha = 1.0 -> only target
    """
    if target_img.shape[:2] != source_img.shape[:2]:
        target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))
    
    blended = cv2.addWeighted(source_img.astype(np.float32), 1-alpha,
                              target_img.astype(np.float32), alpha, 0)
    
    return np.clip(blended, 0, 255).astype(np.uint8)
    
