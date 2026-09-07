from pathlib import Path
from turtle import color
import cv2
import numpy as np
import os
import math
from tqdm import tqdm
from collections import defaultdict
import traceback
from skimage.morphology import skeletonize, medial_axis  # skeletonisation functions
import matplotlib.pyplot as plt


def get_od_max_circle(od_mask):
    """
    Args:
        od_mask (np.ndarray): binary optic disc mask
        
    Returns:
        tuple: 
            - (cx, cy) (tuple[int, int]): centre of the minimum enclosing circle
            - dd (float): optic disc diameter
    """

    contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0), 0.0
    
    max_contour = max(contours, key=cv2.contourArea)

    (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
    dd = 2 * radius
    
    return (int(cx), int(cy)), dd


def generate_annular_masks(av_img, od_center, dd):
    """
    Args:
        av_img (np.ndarray): artery and vein segmentation image
        od_center (tuple[int, int]): coordinates of the optic disc centre
        dd (float): optic disc diameter
        
    Returns:
        tuple: 
            - a_mask (np.ndarray): zone A mask
            - b_mask (np.ndarray): zone B mask
            - c_mask (np.ndarray): zone C mask
    """
    h, w = av_img.shape[:2]
    cx, cy = od_center
    
    a_mask = np.zeros((h, w), dtype=np.uint8)
    b_mask = np.zeros((h, w), dtype=np.uint8)
    c_mask = np.zeros((h, w), dtype=np.uint8)
    
    od_radius = dd / 2
    a_outer_radius = od_radius + 0.5 * dd
    b_outer_radius = od_radius + 1.0 * dd
    c_outer_radius = od_radius + 2.0 * dd
    
    cv2.circle(a_mask, (cx, cy), int(a_outer_radius), 255, -1)
    cv2.circle(a_mask, (cx, cy), int(od_radius), 0, -1)
    
    cv2.circle(b_mask, (cx, cy), int(b_outer_radius), 255, -1)
    cv2.circle(b_mask, (cx, cy), int(a_outer_radius), 0, -1)
    
    cv2.circle(c_mask, (cx, cy), int(c_outer_radius), 255, -1)
    cv2.circle(c_mask, (cx, cy), int(b_outer_radius), 0, -1)
    
    return a_mask, b_mask, c_mask


def get_top_n_vessels_in_c(vessel_mask, c_mask, top_n=6):
    """
    Args:
        vessel_mask (np.ndarray): binary vessel mask (uint8)
        c_mask (np.ndarray): zone C mask (uint8)
        top_n (int): number of widest vessel segments to select

    Returns:
        list[float]: maximum diameters of the N widest segments, in descending order
    """

    vessel_in_c = cv2.bitwise_and(vessel_mask, vessel_mask, mask=c_mask)

    _, bin_mask = cv2.threshold(vessel_in_c, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8, cv2.CV_32S)

    # The maximum diameter is computed separately for each connected component
    # (vessel segment), and the top_n widest are then selected by diameter.
    # The challenge definition refers to the "six widest segments", so the
    # selection is made on diameter (width) rather than on area.
    diameters = []
    for i in range(1, num_labels):
        # Extract this segment exactly using the label rather than a bounding
        # box, so that neighbouring vessels are not included
        seg_mask = (labels == i).astype(np.uint8) * 255
        skeleton, dist = medial_axis(seg_mask, return_distance=True)
        if not skeleton.any():
            continue
        seg_diameters = dist[skeleton] * 2
        diameters.append(float(seg_diameters.max()))

    diameters.sort(reverse=True)
    diameters = diameters[:top_n]

    if len(diameters) < top_n:
        diameters += [0.0] * (top_n - len(diameters))
    return diameters


def calculate_crae_crve_revised(vessel_areas, is_artery = True):
    """
    Args:
        vessel_areas (list[float]): diameters of the six widest segments within zone C
        is_artery (bool): True computes CRAE, False computes CRVE

    Returns:
        float: the resulting CRAE or CRVE value
    """

    coeff = 0.88 if is_artery else 0.95

    values = sorted(vessel_areas, reverse=True)

    while len(values) > 1:
        values = sorted(values, reverse=True)
        next_values = []
        i = 0
        j = len(values) - 1

        while i < j:
            w1 = values[i]
            w2 = values[j]
            w_new = coeff * math.sqrt(w1 ** 2 + w2 ** 2)
            next_values.append(w_new)
            i += 1
            j -= 1


        if i == j:
            next_values.append(values[i])

        values = next_values

    return values[0]


def calculate_density_in_c(vessel_mask, c_mask):
    """
    Args:
        vessel_mask (np.ndarray): binary vessel mask
        c_mask (np.ndarray): zone C mask
        
    Returns:
        float: vessel density
    """

    _, vessel_bin = cv2.threshold(vessel_mask, 127, 1, cv2.THRESH_BINARY)
    _, c_bin = cv2.threshold(c_mask, 127, 1, cv2.THRESH_BINARY)
    
    vessel_in_c = vessel_bin * c_bin
    vessel_pixels = np.sum(vessel_in_c)
    c_pixels = np.sum(c_bin)
    
    if c_pixels == 0:
        return 0.0
    
    return vessel_pixels / c_pixels


def calculate_fractal_dimension_skeleton(binary_img):
    """
    Args:
        binary_img (np.ndarray): binary vessel mask (uint8)
        
    Returns:
        float: fractal dimension
    """
    
    if binary_img.max() == 0:
        return 0.0
    
    _, binary = cv2.threshold(binary_img, 127, 1, cv2.THRESH_BINARY)
    
    skeleton = skeletonize(binary).astype(np.uint8)

    rows, cols = skeleton.shape
    max_box_size = min(rows, cols) // 2
    min_box_size = 1
    
    box_sizes = []
    box_counts = []
    #box_size = min_box_size
    
    for box_size in range(min_box_size, max_box_size + 1):
        
        count = 0

        for i in range(0, rows, box_size):
            for j in range(0, cols, box_size):

                i_end = min(i + box_size, rows)
                j_end = min(j + box_size, cols)
                
                if np.sum(skeleton[i:i_end, j:j_end]) > 0:
                    count += 1
        
        if count > 0:
            box_sizes.append(math.log(1.0 / box_size))
            box_counts.append(math.log(count))
       
    if len(box_sizes) < 2:
        return 0.0
    
    coeffs = np.polyfit(box_sizes, box_counts, 1)

    return coeffs[0]


def extract_av_masks(av_img):
    """
    Args:
        av_img (np.ndarray): RGB artery and vein segmentation image
        
    Returns:
        tuple:
            - artery_mask (np.ndarray): binary artery mask
            - vein_mask (np.ndarray): binary vein mask
    """
    
    r_channel = av_img[:, :, 0]
    g_channel = av_img[:, :, 1]
    b_channel = av_img[:, :, 2]
    
    artery_mask = np.logical_and(g_channel, ~b_channel).astype(np.uint8) * 255
    vein_mask = np.logical_and(g_channel, ~r_channel).astype(np.uint8) * 255
    return artery_mask, vein_mask


def process_av_indicators(av_dir, disc_dir, output_dir):
    """
    Compute the seven artery and vein biomarkers and save the results.

    Biomarkers:
    1. CRAE - central retinal artery equivalent
    2. CRVE - central retinal vein equivalent
    3. AVR  - arteriolar-to-venular ratio (CRAE/CRVE)
    4. artery_density - artery density within zone C
    5. vein_density - vein density within zone C
    6. artery_fractal_dimension - artery fractal dimension
    7. vein_fractal_dimension - vein fractal dimension
    
    Args:
        av_dir (str): directory of artery and vein segmentation images
        disc_dir (str): directory of optic disc mask images
        output_dir (str): directory in which the result files are saved
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    av_suffixes = ('.png', '.PNG')
    av_files = [f for f in os.listdir(av_dir) if f.lower().endswith(av_suffixes)]
    
    for fname in tqdm(av_files, desc="Calculating AV indicators"):
        try:
            av_path = os.path.join(av_dir, fname)
            disc_path = os.path.join(disc_dir, fname)
            txt_path = os.path.join(output_dir, Path(fname).stem + '.txt')
            
            if not os.path.exists(disc_path):
                print(f"Warning: Disc file {fname} not found, skip")
                continue
            
            av_img = cv2.imread(av_path, cv2.IMREAD_COLOR)
            av_img = cv2.cvtColor(av_img, cv2.COLOR_BGR2RGB)
            disc_img = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
            
            if av_img is None or disc_img is None:
                print(f"Warning: Failed to read {fname}, skip")
                continue
            
            if av_img.shape[:2] != disc_img.shape:
                disc_img = cv2.resize(disc_img, (av_img.shape[1], av_img.shape[0]))
            
            artery_mask, vein_mask = extract_av_masks(av_img)
            
            _, od_bin = cv2.threshold(disc_img, 200, 255, cv2.THRESH_BINARY)
            
            od_center, dd = get_od_max_circle(od_bin)
            if dd == 0:
                print(f"Warning: OD circle not found for {fname}, skip")
                continue
            
            _, _, c_mask = generate_annular_masks(av_img, od_center, dd)
            
            top6_artery_areas = get_top_n_vessels_in_c(artery_mask, c_mask, top_n=6)
            top6_vein_areas = get_top_n_vessels_in_c(vein_mask, c_mask, top_n=6)
            
            crae = calculate_crae_crve_revised(top6_artery_areas, is_artery = True)
            crve = calculate_crae_crve_revised(top6_vein_areas, is_artery = False)
            
            avr = crae / crve if crve > 0 and crae > 0 else float('inf')
            
            artery_density = calculate_density_in_c(artery_mask, c_mask)
            vein_density = calculate_density_in_c(vein_mask, c_mask)
            
            artery_fractal = calculate_fractal_dimension_skeleton(artery_mask)
            vein_fractal = calculate_fractal_dimension_skeleton(vein_mask)
            
            results = {
                "CRAE": crae,
                "CRVE": crve,
                "AVR": avr,
                "artery_density": artery_density,
                "vein_density": vein_density,
                "artery_fractal_dimension": artery_fractal,
                "vein_fractal_dimension": vein_fractal
            }
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for idx, (key, value) in enumerate(results.items(), 1):
                    if isinstance(value, float):
                        if math.isinf(value):
                            f.write(f"{key} N/A (zero denominator)\n")
                        else:
                            f.write(f"{key} {value:.6f}\n")
                    else:
                        f.write(f"{key} {value}\n")
            
            print(f"Successfully saved results to {txt_path}")
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            print(traceback.format_exc())
            continue



if __name__ == "__main__":
    
    AV_DIR = r"./results/weighted_av"
    
    DISC_DIR = r"./data/training/disc_traditional"
    OUTPUT_DIR = r"./results/biomarker_weighted"
    process_av_indicators(AV_DIR, DISC_DIR, OUTPUT_DIR)