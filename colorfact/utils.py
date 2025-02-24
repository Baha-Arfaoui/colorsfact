import cv2
import numpy as np
import requests
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import time
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()


class LoadToolsConfig:

    def __init__(self) -> None:
        with open(here("colorfact/configs/config.yaml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Set environment variables
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
      

        # Faiss Index
        self.faiss_index = app_config["faiss_index"]["path"]
     


class extract_colors_url():
    def __init__(self, image_url):
        self.image_url = image_url
        self.colors = self.extract_dominant_color_from_segmented()

    def read_image_from_url(self):
        """
        Downloads an image from a URL and returns it as a NumPy array in BGR format.
        """
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}  # Mimic a browser request
            response = requests.get(self.image_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an error if request fails

            try:
                image = Image.open(BytesIO(response.content))
                image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode image")
                return image

            except UnidentifiedImageError:
                print(f"Error: Cannot identify image from {self.image_url}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
 

    def remove_background_grabcut(self, rect=None, iter_count=5):
        """
        Uses GrabCut to segment the foreground (product) from the background.
        Downloads the image from a URL. If rect is not provided, a default rectangle is used.
        
        Returns the foreground mask where pixels belonging to the product are 1.
        """
        image = self.read_image_from_url()
        
        # If no rectangle is provided, use a rectangle covering most of the image
        height, width = image.shape[:2]
        if rect is None:
            rect = (int(0.05 * width), int(0.05 * height),
                    int(0.9 * width), int(0.9 * height))
        
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
        # Define foreground as either sure or probable foreground
        mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
        return mask2

    def convert_opencv_lab_to_cielab(self,lab_pixel):
        """
        Convert an OpenCV LAB pixel (0-255 for L, centered at 128 for a and b)
        to the standard CIELAB scale (L: 0-100, a: -128 to 128, b: -128 to 128).
        """

        L_opencv, a_opencv, b_opencv = lab_pixel
        L_std =  ((L_opencv / 255)*100)
        a_std = float(a_opencv - 128)
        b_std = float(b_opencv - 128)
        return np.array([L_std, a_std, b_std])

    def process_floats(self,color_list: list,prop_list: list[float]) -> float | list[float]:
        """
        Processes a list of three floats according to the specified conditions.

        Args:
            float_list: A list of three floats.

        Returns:
            The first item if it's greater than 85, the sum of the first two items if 
            their sum is greater than 90, or the original list if neither condition is met.
        """

        if not isinstance(prop_list, list) or len(prop_list) != 3 or not all(isinstance(x, (int, float)) for x in prop_list):
            raise ValueError("Input must be a list of three numbers.")

        first_item = float(prop_list[0])
        second_item = float(prop_list[1])
    
        if (first_item > 0.7):  # Changed condition to use 0.85 instead of 85
            return [color_list[0]]
        elif (first_item + second_item > 0.8):  # Changed condition to use 0.9 instead of 90
            return [color_list[0],color_list[1]]
        else:
            return color_list
    
    def extract_dominant_color_from_segmented(self, k=3, resize_dim=(100,100)):
        """
        Downloads the image from a URL, resizes it, applies the provided mask
        to extract foreground pixels, runs k-means clustering on these pixels,
        and returns a list of dominant colors (in standard CIELAB space) along with their proportions.
        """
        mask=self.remove_background_grabcut()
        image = self.read_image_from_url()
        image = cv2.resize(image, resize_dim)
        mask = cv2.resize(mask, resize_dim)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use only pixels where mask == 1
        pixels = image_rgb.reshape(-1, 3)
        mask = mask.flatten().astype(bool)
        pixels_fg = pixels[mask]
        
        # Run k-means clustering on foreground pixels
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels_fg)
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Sort clusters by frequency (most common first)
        sorted_idx = sorted(counts, key=counts.get, reverse=True)
        
        sorted_lab_colors = []
        proportions = []
        for idx in sorted_idx:
            centroid_rgb = kmeans.cluster_centers_[idx]
            centroid_rgb_uint8 = np.uint8([[centroid_rgb]])
            # Convert from RGB to LAB using OpenCV's conversion
            lab = cv2.cvtColor(centroid_rgb_uint8, cv2.COLOR_RGB2LAB)[0][0]
        
            lab_std = self.convert_opencv_lab_to_cielab(lab)
        
            sorted_lab_colors.append(lab_std)
            proportions.append(counts[idx] / len(pixels_fg))
        
        sorted_lab_colors=[item.tolist() for item in sorted_lab_colors]
        
        color=self.process_floats(sorted_lab_colors,proportions)
        return color
    




class extract_colors_image():
    def __init__(self, image_path):
        self.image_path = image_path
        self.colors = self.extract_dominant_color_from_segmented()

    def read_image_from_local(self):
        """
        Reads an image from a local path and returns it as a NumPy array in BGR format.
        """
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Failed to read image from {self.image_path}")
            return image
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

    def remove_background_grabcut(self, rect=None, iter_count=5):
        """
        Uses GrabCut to segment the foreground (product) from the background.
        If rect is not provided, a default rectangle is used.
        
        Returns the foreground mask where pixels belonging to the product are 1.
        """
        image = self.read_image_from_local()
        
        # If no rectangle is provided, use a rectangle covering most of the image
        height, width = image.shape[:2]
        if rect is None:
            rect = (int(0.05 * width), int(0.05 * height),
                    int(0.9 * width), int(0.9 * height))
        
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
        # Define foreground as either sure or probable foreground
        mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
        return mask2

    def convert_opencv_lab_to_cielab(self, lab_pixel):
        """
        Convert an OpenCV LAB pixel (0-255 for L, centered at 128 for a and b)
        to the standard CIELAB scale (L: 0-100, a: -128 to 128, b: -128 to 128).
        """
        L_opencv, a_opencv, b_opencv = lab_pixel
        L_std =  ((L_opencv / 255)*100)
        a_std = float(a_opencv - 128)
        b_std = float(b_opencv - 128)
        return np.array([L_std, a_std, b_std])

    def process_floats(self, color_list: list, prop_list: list[float]) -> float | list[float]:
        """
        Processes a list of three floats according to the specified conditions.

        Args:
            float_list: A list of three floats.

        Returns:
            The first item if it's greater than 85, the sum of the first two items if 
            their sum is greater than 90, or the original list if neither condition is met.
        """
        if not isinstance(prop_list, list) or len(prop_list) != 3 or not all(isinstance(x, (int, float)) for x in prop_list):
            raise ValueError("Input must be a list of three numbers.")

        first_item = float(prop_list[0])
        second_item = float(prop_list[1])
    
        if (first_item > 0.7):  # Changed condition to use 0.85 instead of 85
            return [color_list[0]]
        elif (first_item + second_item > 0.8):  # Changed condition to use 0.9 instead of 90
            return [color_list[0], color_list[1]]
        else:
            return color_list

    def extract_dominant_color_from_segmented(self, k=3, resize_dim=(100, 100)):
        """
        Reads the image from a local path, resizes it, applies the provided mask
        to extract foreground pixels, runs k-means clustering on these pixels,
        and returns a list of dominant colors (in standard CIELAB space) along with their proportions.
        """
        mask = self.remove_background_grabcut()
        image = self.read_image_from_local()
        image = cv2.resize(image, resize_dim)
        mask = cv2.resize(mask, resize_dim)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use only pixels where mask == 1
        pixels = image_rgb.reshape(-1, 3)
        mask = mask.flatten().astype(bool)
        pixels_fg = pixels[mask]
        
        # Run k-means clustering on foreground pixels
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels_fg)
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Sort clusters by frequency (most common first)
        sorted_idx = sorted(counts, key=counts.get, reverse=True)
        
        sorted_lab_colors = []
        proportions = []
        for idx in sorted_idx:
            centroid_rgb = kmeans.cluster_centers_[idx]
            centroid_rgb_uint8 = np.uint8([[centroid_rgb]])
            # Convert from RGB to LAB using OpenCV's conversion
            lab = cv2.cvtColor(centroid_rgb_uint8, cv2.COLOR_RGB2LAB)[0][0]
        
            lab_std = self.convert_opencv_lab_to_cielab(lab)
        
            sorted_lab_colors.append(lab_std)
            proportions.append(counts[idx] / len(pixels_fg))
        
        sorted_lab_colors = [item.tolist() for item in sorted_lab_colors]
        
        color = self.process_floats(sorted_lab_colors, proportions)
        return color
    


class matching_products():

    def __init__(self, faiss_index,data,ontologie):
        self.faiss_index=faiss_index
        self.data=data
        self.ontologie=ontologie

    def lab_to_lch(self,lab):
        L, a, b = lab
        C = np.sqrt(a**2 + b**2)
        H = np.degrees(np.arctan2(b, a)) % 360  # Normalisation correcte
        return [L, C, H]

    def lch_to_lab(self,lch):
        L, C, H = lch
        a = C * np.cos(np.radians(H))
        b = C * np.sin(np.radians(H))
        return [L, a, b]

    def generate_harmonic_colors(self,input_colors_lab):
    
        input_colors_lch = [self.lab_to_lch(color) for color in input_colors_lab]
        num_colors = len(input_colors_lab)

        if num_colors == 1:
            L, C, H = input_colors_lch[0]
            triad1 = self.lch_to_lab([L, C, (H + 120) % 360])
            triad2 = self.lch_to_lab([L, C, (H + 240) % 360])
            return input_colors_lab + [triad1, triad2]

        elif num_colors == 2:
            chosen_idx = np.random.choice([0, 1])
            L, C, H = input_colors_lch[chosen_idx]
            complementary = self.lch_to_lab([L, C, (H + 180) % 360])
            return input_colors_lab + [complementary]

        elif num_colors == 3:
            chosen_indices = np.random.choice([0, 1, 2], size=2, replace=False)
            selected_colors = [input_colors_lab[i] for i in chosen_indices]
            return selected_colors

        return input_colors_lab


    def recommend_outfit(self,input_category, input_colors, outfit_type, faiss_index, df, ontology, top_k=5, min_items=1):

        try:

            required_items = ontology["Outfit"][input_category][outfit_type]

        except KeyError:

            raise ValueError(f"Invalid category '{input_category}' or outfit type '{outfit_type}'")



        recommendations = {}

        

        for item_category in required_items:

            # Step 1: Search globally (without filtering first)
            input_colors = self.generate_harmonic_colors(input_colors)

            n_neighbors = top_k * 3  # Overfetch to account for category filtering

            distances, indices = faiss_index.search(np.array(input_colors, dtype="float32"), n_neighbors)

            

            # Step 2: Map indices to product IDs and filter by category

            seen_ids = set()

            item_matches = []

            

            # Flatten all indices/distances across input colors

            for color_idx in range(len(input_colors)):

                for rank_idx in range(n_neighbors):

                    global_idx = indices[color_idx][rank_idx]

                    product_id = df.iloc[global_idx]["Photo produit 1"]

                    product_row = df.iloc[global_idx]

                    

                    # Check if product matches the target category (case-insensitive)

                    if product_row["Catégorie produit"].lower() == item_category.lower():

                        if product_id not in seen_ids:

                            seen_ids.add(product_id)

                            item_matches.append({

                                "product_id": product_id,

                                "distance": distances[color_idx][rank_idx],

                                "metadata": product_row.to_dict()

                            })

                            # Break early if we have enough matches

                            if len(item_matches) >= top_k:

                                break

            

            # Step 3: Limit to top_k and handle min_items

            final_matches = item_matches[:top_k]

            if len(final_matches) < min_items:

                print(f"Warning: Only {len(final_matches)} {item_category} found (min_items={min_items})")

            

            recommendations[item_category] = final_matches

        

        return recommendations