import pandas as pd
import numpy as np
import pydicom as dicom
import os
import png
import time
import cv2


from dotenv import load_dotenv


def get_cancer_df(threshold_for_cancer):
    """Generate a dataframe containing information on what is cancer
    and what the path to the associated files for only US photos.

    Args:
        threshold_for_cancer (int, optional): The UCLA threadhold for cancer.
            Anything above this given value will be marked as cancer.

    Returns:
        pd.DataFrame:DataFrame containing results of what is cancer.
    """
    # Grab required information from the metadatacsv file.
    metadata_df = pd.read_csv("datasets/metadata.csv")
    meta_data_columns_to_keep = [
        "Subject ID",
        "Modality",
        "SOP Class Name",
        "File Location",
        "Series UID",
    ]
    metadata_df = metadata_df[meta_data_columns_to_keep]
    metadata_df = metadata_df[metadata_df["Modality"] == "US"]

    # Grab required information from the target data excel file.
    target_data_df = pd.read_excel(
        "datasets/Target Data_2019-12-05.xlsx", sheet_name="Sheet1"
    )
    target_columns_to_keep = [
        "UCLA Score (Similar to PIRADS v2)",
        "Patient ID",
        "seriesInstanceUID_US",
    ]
    target_data_df = target_data_df[target_columns_to_keep]
    target_data_df.rename(columns={"seriesInstanceUID_US": "Series UID"}, inplace=True)

    # Determine if things in target df are cancerous.
    target_data_df["cancer"] = (
        target_data_df["UCLA Score (Similar to PIRADS v2)"] >= threshold_for_cancer
    )

    # Merge the 2 infomation sources together to give us what we need to know
    # for what what files are and are not cancerous.
    cancer_df = pd.merge(metadata_df, target_data_df, on="Series UID")

    return cancer_df


def create_image_datasets(cancer_df, path_to_manifest_folder):
    """Creates image datasets for cancer and nonmalignant.

    Args:
        cancer_df (pd.DataFrame): Contains location of images to add to dataset.
        path_to_manifest_folder (str): Location of the manifest folder.
    """
    cancer_image_path = "images/cancer"
    if not os.path.exists(cancer_image_path):
        os.makedirs(cancer_image_path)

    nonmalignant_image_path = "images/nonmalignant"
    if not os.path.exists(nonmalignant_image_path):
        os.makedirs(nonmalignant_image_path)

    for i in range(cancer_df.shape[0]):
        current_location = os.path.join(os.getcwd())

        location_of_image = cancer_df.iloc[i]["File Location"][2:]

        original_image_path = os.path.join(path_to_manifest_folder, location_of_image)

        try:
            name_of_dcm_file = os.listdir(original_image_path)[0]
        except Exception:
            print(f"Unable to work with file: {original_image_path}; moving on...")
            continue

        if cancer_df.iloc[i]["cancer"]:
            new_image_location = os.path.join(
                current_location, cancer_image_path
            )
        else:
            new_image_location = os.path.join(
                current_location, nonmalignant_image_path
            )

        original_image_path = os.path.join(original_image_path, name_of_dcm_file)
        ds = dicom.dcmread(original_image_path)
        pixel_array = ds.pixel_array.astype(float)

        for slice_number in range(pixel_array.shape[0]):
            # We only care about things in this range.
            if slice_number not in [130, 134]:
                continue

            slice_image_path = os.path.join(
                new_image_location,
                f"image_{i}_{slice_number}.jpg"
            )

            if cv2.imwrite(slice_image_path, pixel_array[slice_number]):
                print(f"{new_image_location} created successfully\n")
            else:
                print(f"Failed to create {new_image_location}...\n")


def main():

    # Load environment variables into project.
    load_dotenv("settings.env")

    # Settings for this project. These can be changed in the settings.env file.
    ucla_cancer_threshold = int(os.getenv("UCLA_CANCER_THRESHOLD"))
    path_to_manifest_folder = os.getenv("PATH_TO_MANIFEST_FOLDER")

    print("Generating information on cancer/noncancerous photos...")
    cancer_df = get_cancer_df(threshold_for_cancer=ucla_cancer_threshold)

    print("Dataset distribution (msg shown for 5 seconds)...")
    print(cancer_df["cancer"].value_counts())
    time.sleep(5)

    print("Creating dataset now (msg shown for 5 seconds)...")
    time.sleep(5)
    create_image_datasets(
        cancer_df=cancer_df, path_to_manifest_folder=path_to_manifest_folder
    )


if __name__ == "__main__":
    main()
