import pandas as pd
import argparse
from utils import extract_colors_image, extract_colors_url
import concurrent.futures
from tqdm import tqdm
import ast

def process_image(image_url):
    """Extracts dominant colors from an image URL."""
    try:
        color_extractor = extract_colors_url(image_url)
        return color_extractor.colors
    except Exception as e:
        print(f"Error processing {image_url}: {str(e)}")
        return None

def process_dataset(df: pd.DataFrame, num_workers=8) -> pd.DataFrame:
    """
    Processes a dataset containing image links and adds a 'cielab_colors' column
    with the dominant colors extracted from each image using multithreading.
    """
    if 'Photo produit 1' not in df.columns:
        raise ValueError("DataFrame must contain a 'Photo produit 1' column")

    image_urls = df['Photo produit 1'].tolist()

    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image, image_urls), total=len(image_urls)))

    df['cielab_colors'] = results
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-O", "--output", required=True, help="Path to the input dataset file")
    args = parser.parse_args()

    data = pd.read_excel(args.output)
    
    try:
        data = data.drop("Unnamed: 0", axis=1)
    except KeyError:
        pass
    
    data_processed = process_dataset(data, num_workers=8)

    def safe_literal_eval(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val  # If already a list, return as is

    data_processed["cielab_colors"] = data_processed["cielab_colors"].apply(safe_literal_eval)
    
    data_processed = data_processed.explode("cielab_colors").reset_index(drop=True)
    data_processed=data_processed.drop("Prix ",axis=1)

    # Join with existing dataset
    data_output = pd.read_excel("data/output.xlsx", index_col=0)
    dataset = pd.concat([data_output, data_processed], ignore_index=True)
    dataset["cielab_colors"] = dataset["cielab_colors"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    dataset = dataset.drop_duplicates()

    
    dataset.to_excel("data/new_output.xlsx")