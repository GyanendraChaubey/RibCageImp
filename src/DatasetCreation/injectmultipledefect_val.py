import os
import glob
import nibabel as nib
import numpy as np
import random

def generate_chest_defect(data, cube_dim):
    x_, y_, z_ = data.shape
    chest_start = int(z_ * 0.5)  # Starting at the middle of the z-dimension, assuming the chest is in the upper half
    chest_end = int(z_ * 0.75)   # Extend to three-quarters of the z-dimension

    full_masking = np.ones(shape=(x_, y_, z_))
    x = random.randint(int(cube_dim / 2), x_ - int(cube_dim / 2))
    y = random.randint(int(cube_dim / 2), y_ - int(cube_dim / 2))
    z = random.randint(chest_start, chest_end)

    cube_masking = np.zeros(shape=(cube_dim, cube_dim, z_ - z))
    full_masking[x - int(cube_dim / 2):x + int(cube_dim / 2),
                 y - int(cube_dim / 2):y + int(cube_dim / 2),
                 z:z_] = cube_masking

    # defected data
    defected_data = full_masking * data
    # implant data
    implant_data = (1 - full_masking) * data

    return defected_data, implant_data

def process_segmented_files(file_path, cube_dim, defected_dir, implant_dir, num_defects=5):
    """Process a single .nii file to introduce multiple defects and save them, along with the extracted defects."""
    nii_img = nib.load(file_path)
    data = nii_img.get_fdata()
    affine = nii_img.affine
    header = nii_img.header

    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)

    for i in range(num_defects):
        defected_data, implant_data = generate_chest_defect(data, cube_dim)

        # Save the defected data and implant
        defected_path = os.path.join(defected_dir, f"{base_name[:-10]}_defected_{i+1}.nii.gz")
        implant_path = os.path.join(implant_dir, f"{base_name[:-10]}_implant_{i+1}.nii.gz")

        defected_img = nib.Nifti1Image(defected_data, affine, header)
        nib.save(defected_img, defected_path)
        print(f"Processed and saved defected: {defected_path}")

        implant_img = nib.Nifti1Image(implant_data, affine, header)
        nib.save(implant_img, implant_path)
        print(f"Processed and saved implant: {implant_path}")

def main(segmented_dir, defected_dir, implant_dir, cube_dim, num_defects=5):
    """Main function to process all segmented .nii files to introduce multiple defects and save implants."""
    nii_files = glob.glob(os.path.join(segmented_dir, '*.nii')) + glob.glob(os.path.join(segmented_dir, '*.nii.gz'))
    os.makedirs(defected_dir, exist_ok=True)
    os.makedirs(implant_dir, exist_ok=True)

    for nii_file in nii_files:
        print(f"Processing file: {nii_file}")
        process_segmented_files(nii_file, cube_dim, defected_dir, implant_dir, num_defects)

    print(f"Processing complete. Results saved in: {defected_dir} and {implant_dir}")

if __name__ == "__main__":
    # Directory containing the segmented files
    segmented_dir = '/workspace/RibCage/val-segmented_ribfrac'  
    defected_dir = '/workspace/RibCage/val-ribfrac-defected'  
    implant_dir = '/workspace/RibCage/val-ribfrac-implants'  
    cube_dim = 64 
    num_defects = 5  # Number of defects to generate for each file

    main(segmented_dir, defected_dir, implant_dir, cube_dim, num_defects)