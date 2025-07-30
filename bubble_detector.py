import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class BubblePhotoGenerator:
    def __init__(self, filled_threshold: float = 0.6, min_radius: int = 8, max_radius: int = 25):
        """
        Initialize the Bubble Photo Generator
        
        Args:
            filled_threshold: Threshold ratio to determine if a bubble is filled (0.0-1.0)
            min_radius: Minimum radius for bubble detection
            max_radius: Maximum radius for bubble detection
        """
        self.filled_threshold = filled_threshold
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def detect_circles(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect circles/bubbles in the image using HoughCircles
        
        Args:
            image: Input grayscale image
            
        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Use HoughCircles to detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(self.min_radius * 1.5),
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_circles.append((x, y, r))
        
        return detected_circles
    
    def is_bubble_filled(self, image: np.ndarray, x: int, y: int, radius: int) -> bool:
        """
        Determine if a bubble is filled by analyzing the pixel intensity within the circle
        
        Args:
            image: Grayscale image
            x, y: Center coordinates of the bubble
            radius: Radius of the bubble
            
        Returns:
            True if bubble is filled, False otherwise
        """
        # Create a mask for the circle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius - 2, 255, -1)  # Slightly smaller radius to avoid border
        
        # Extract pixel values within the circle
        circle_pixels = image[mask == 255]
        
        if len(circle_pixels) == 0:
            return False
        
        # Calculate the ratio of dark pixels (assuming filled bubbles are darker)
        dark_pixel_count = np.sum(circle_pixels < 128)  # Threshold for "dark"
        total_pixels = len(circle_pixels)
        dark_ratio = dark_pixel_count / total_pixels
        
        return dark_ratio > self.filled_threshold
    
    def generate_bubble_photo(self, image_path: str, save_path: str = None) -> np.ndarray:
        """
        Generate the bubble photo according to specifications:
        - Black background everywhere
        - White circles for unfilled bubbles
        - Green circles for filled bubbles
        
        Args:
            image_path: Path to the input OMR sheet image
            save_path: Optional path to save the output image
            
        Returns:
            The generated bubble photo as numpy array
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create output image (black background)
        height, width = gray.shape
        bubble_photo = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Detect all circles/bubbles
        print("Detecting bubbles...")
        circles = self.detect_circles(gray)
        print(f"Found {len(circles)} potential bubbles")
        
        # Process each detected circle
        filled_count = 0
        unfilled_count = 0
        
        for x, y, r in circles:
            # Check if the bubble is filled
            is_filled = self.is_bubble_filled(gray, x, y, r)
            
            if is_filled:
                # Draw green circle for filled bubbles
                cv2.circle(bubble_photo, (x, y), r, (0, 255, 0), -1)  # Green (BGR format)
                filled_count += 1
            else:
                # Draw white circle for unfilled bubbles
                cv2.circle(bubble_photo, (x, y), r, (255, 255, 255), -1)  # White
                unfilled_count += 1
        
        print(f"Filled bubbles: {filled_count}")
        print(f"Unfilled bubbles: {unfilled_count}")
        
        # Save the result if path provided
        if save_path:
            cv2.imwrite(save_path, bubble_photo)
            print(f"Bubble photo saved to: {save_path}")
        
        return bubble_photo
    
    def generate_bubble_photo_advanced(self, image_path: str, save_path: str = None, 
                                     show_debug: bool = False) -> np.ndarray:
        """
        Advanced version with better bubble detection and preprocessing
        
        Args:
            image_path: Path to the input OMR sheet image
            save_path: Optional path to save the output image
            show_debug: Whether to show debug visualizations
            
        Returns:
            The generated bubble photo as numpy array
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing to enhance bubble detection
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Noise reduction
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Create output image (black background)
        height, width = gray.shape
        bubble_photo = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Try multiple parameter sets for circle detection to catch more bubbles
        all_circles = []
        
        # Parameter set 1: Standard detection
        circles1 = self._detect_circles_with_params(denoised, param1=50, param2=30)
        all_circles.extend(circles1)
        
        # Parameter set 2: More sensitive detection
        circles2 = self._detect_circles_with_params(denoised, param1=40, param2=25)
        all_circles.extend(circles2)
        
        # Parameter set 3: Less sensitive but more accurate
        circles3 = self._detect_circles_with_params(denoised, param1=60, param2=35)
        all_circles.extend(circles3)
        
        # Remove duplicate circles (those that are very close to each other)
        unique_circles = self._remove_duplicate_circles(all_circles)
        
        print(f"Total circles detected: {len(all_circles)}")
        print(f"Unique circles after deduplication: {len(unique_circles)}")
        
        # Process each unique circle
        filled_count = 0
        unfilled_count = 0
        
        for x, y, r in unique_circles:
            # Use multiple methods to determine if bubble is filled
            is_filled = self._is_bubble_filled_advanced(gray, x, y, r)
            
            if is_filled:
                # Draw green circle for filled bubbles
                cv2.circle(bubble_photo, (x, y), r, (0, 255, 0), -1)  # Green
                filled_count += 1
            else:
                # Draw white circle for unfilled bubbles
                cv2.circle(bubble_photo, (x, y), r, (255, 255, 255), -1)  # White
                unfilled_count += 1
        
        print(f"Filled bubbles: {filled_count}")
        print(f"Unfilled bubbles: {unfilled_count}")
        
        # Show debug information if requested
        if show_debug:
            self._show_debug_info(image, gray, denoised, unique_circles, bubble_photo)
        
        # Save the result if path provided
        if save_path:
            cv2.imwrite(save_path, bubble_photo)
            print(f"Bubble photo saved to: {save_path}")
        
        return bubble_photo
    
    def _detect_circles_with_params(self, image: np.ndarray, param1: int, param2: int) -> List[Tuple[int, int, int]]:
        """Helper method to detect circles with specific parameters"""
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(self.min_radius * 1.2),
            param1=param1,
            param2=param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_circles.append((x, y, r))
        
        return detected_circles
    
    def _remove_duplicate_circles(self, circles: List[Tuple[int, int, int]], 
                                overlap_threshold: float = 0.7) -> List[Tuple[int, int, int]]:
        """Remove duplicate/overlapping circles"""
        if not circles:
            return []
        
        # Sort by radius (larger circles first)
        circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)
        unique_circles = []
        
        for x, y, r in circles_sorted:
            is_duplicate = False
            
            for ux, uy, ur in unique_circles:
                # Calculate distance between centers
                distance = np.sqrt((x - ux)**2 + (y - uy)**2)
                # Check if circles overlap significantly
                if distance < (r + ur) * overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_circles.append((x, y, r))
        
        return unique_circles
    
    def _is_bubble_filled_advanced(self, image: np.ndarray, x: int, y: int, radius: int) -> bool:
        """Advanced method to determine if a bubble is filled"""
        # Method 1: Basic darkness ratio
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, radius - 3), 255, -1)
        
        circle_pixels = image[mask == 255]
        if len(circle_pixels) == 0:
            return False
        
        # Calculate statistics
        mean_intensity = np.mean(circle_pixels)
        dark_ratio = np.sum(circle_pixels < 100) / len(circle_pixels)
        
        # Method 2: Compare with surrounding area (ring around the circle)
        outer_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), radius + 5, 255, -1)
        cv2.circle(outer_mask, (x, y), radius + 2, 0, -1)
        
        ring_pixels = image[outer_mask == 255]
        if len(ring_pixels) > 0:
            ring_mean = np.mean(ring_pixels)
            intensity_difference = ring_mean - mean_intensity
        else:
            intensity_difference = 0
        
        # Decision logic: bubble is filled if it's significantly darker than surroundings
        # and has a high ratio of dark pixels
        is_filled = (dark_ratio > self.filled_threshold) or \
                   (intensity_difference > 30 and dark_ratio > 0.4)
        
        return is_filled
    
    def _show_debug_info(self, original: np.ndarray, gray: np.ndarray, 
                        enhanced: np.ndarray, circles: List[Tuple[int, int, int]], 
                        result: np.ndarray):
        """Show debug visualizations"""
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(1, 5, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Grayscale
        plt.subplot(1, 5, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        # Enhanced
        plt.subplot(1, 5, 3)
        plt.imshow(enhanced, cmap='gray')
        plt.title('Enhanced')
        plt.axis('off')
        
        # Detected circles overlay
        circles_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, r in circles:
            cv2.circle(circles_overlay, (x, y), r, (0, 255, 0), 2)
            cv2.circle(circles_overlay, (x, y), 2, (0, 0, 255), 3)
        
        plt.subplot(1, 5, 4)
        plt.imshow(cv2.cvtColor(circles_overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Circles ({len(circles)})')
        plt.axis('off')
        
        # Final result
        plt.subplot(1, 5, 5)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Bubble Photo')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_combined_bubble_photo(self, image_path: str, save_path: str = None,
                                     show_debug: bool = False) -> np.ndarray:
        """
        Generate bubble photo by combining results from both simple and advanced methods
        
        Args:
            image_path: Path to the input OMR sheet image
            save_path: Optional path to save the output image
            show_debug: Whether to show debug visualizations
            
        Returns:
            The combined bubble photo as numpy array
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Get results from both methods
        print("Running simple bubble detection...")
        simple_circles = self.detect_circles(gray)
        
        print("Running advanced bubble detection...")
        # For advanced method, we need to replicate the preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Get circles from advanced method
        all_circles = []
        circles1 = self._detect_circles_with_params(denoised, param1=50, param2=30)
        all_circles.extend(circles1)
        circles2 = self._detect_circles_with_params(denoised, param1=40, param2=25)
        all_circles.extend(circles2)
        circles3 = self._detect_circles_with_params(denoised, param1=60, param2=35)
        all_circles.extend(circles3)
        advanced_circles = self._remove_duplicate_circles(all_circles)
        
        # Combine and deduplicate all circles from both methods
        combined_circles = simple_circles + advanced_circles
        unique_circles = self._remove_duplicate_circles(combined_circles, overlap_threshold=0.6)
        
        print(f"Simple method found: {len(simple_circles)} circles")
        print(f"Advanced method found: {len(advanced_circles)} circles")
        print(f"Combined unique circles: {len(unique_circles)} circles")
        
        # Create output image (black background)
        bubble_photo = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process each circle with consensus from both methods
        filled_count = 0
        unfilled_count = 0
        
        for x, y, r in unique_circles:
            # Check with both simple and advanced methods
            simple_filled = self.is_bubble_filled(gray, x, y, r)
            advanced_filled = self._is_bubble_filled_advanced(gray, x, y, r)
            
            # Consensus decision: if both methods agree, use that result
            # If they disagree, use the advanced method (it's more sophisticated)
            if simple_filled == advanced_filled:
                is_filled = simple_filled
            else:
                # When in doubt, use advanced method but be more conservative
                is_filled = advanced_filled and self._is_bubble_filled_conservative(gray, x, y, r)
            
            if is_filled:
                # Draw green circle for filled bubbles
                cv2.circle(bubble_photo, (x, y), r, (0, 255, 0), -1)  # Green
                filled_count += 1
            else:
                # Draw white circle for unfilled bubbles
                cv2.circle(bubble_photo, (x, y), r, (255, 255, 255), -1)  # White
                unfilled_count += 1
        
        print(f"Final result - Filled bubbles: {filled_count}")
        print(f"Final result - Unfilled bubbles: {unfilled_count}")
        
        # Show debug information if requested
        if show_debug:
            self._show_combined_debug_info(image, gray, simple_circles, advanced_circles, 
                                         unique_circles, bubble_photo)
        
        # Save the result if path provided
        if save_path:
            cv2.imwrite(save_path, bubble_photo)
            print(f"Combined bubble photo saved to: {save_path}")
        
        return bubble_photo
    
    def _is_bubble_filled_conservative(self, image: np.ndarray, x: int, y: int, radius: int) -> bool:
        """
        More conservative method for filled bubble detection when methods disagree
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, radius - 2), 255, -1)
        
        circle_pixels = image[mask == 255]
        if len(circle_pixels) == 0:
            return False
        
        # Use stricter thresholds for conservative detection
        dark_ratio = np.sum(circle_pixels < 90) / len(circle_pixels)
        mean_intensity = np.mean(circle_pixels)
        
        # Bubble is filled only if it's very dark
        return dark_ratio > 0.7 and mean_intensity < 100
    
    def _show_combined_debug_info(self, original: np.ndarray, gray: np.ndarray,
                                simple_circles: List[Tuple[int, int, int]],
                                advanced_circles: List[Tuple[int, int, int]],
                                combined_circles: List[Tuple[int, int, int]],
                                result: np.ndarray):
        """Show debug visualizations for combined method"""
        plt.figure(figsize=(25, 5))
        
        # Original image
        plt.subplot(1, 6, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Simple method circles
        simple_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, r in simple_circles:
            cv2.circle(simple_overlay, (x, y), r, (255, 0, 0), 2)  # Red for simple
            cv2.circle(simple_overlay, (x, y), 2, (255, 0, 0), 3)
        
        plt.subplot(1, 6, 2)
        plt.imshow(cv2.cvtColor(simple_overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'Simple Method ({len(simple_circles)})')
        plt.axis('off')
        
        # Advanced method circles
        advanced_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, r in advanced_circles:
            cv2.circle(advanced_overlay, (x, y), r, (0, 255, 0), 2)  # Green for advanced
            cv2.circle(advanced_overlay, (x, y), 2, (0, 255, 0), 3)
        
        plt.subplot(1, 6, 3)
        plt.imshow(cv2.cvtColor(advanced_overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'Advanced Method ({len(advanced_circles)})')
        plt.axis('off')
        
        # Combined circles
        combined_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, r in combined_circles:
            cv2.circle(combined_overlay, (x, y), r, (0, 0, 255), 2)  # Blue for combined
            cv2.circle(combined_overlay, (x, y), 2, (0, 0, 255), 3)
        
        plt.subplot(1, 6, 4)
        plt.imshow(cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'Combined Unique ({len(combined_circles)})')
        plt.axis('off')
        
        # All methods overlay
        all_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, r in simple_circles:
            cv2.circle(all_overlay, (x, y), r, (255, 0, 0), 1)  # Red for simple
        for x, y, r in advanced_circles:
            cv2.circle(all_overlay, (x, y), r, (0, 255, 0), 1)  # Green for advanced
        for x, y, r in combined_circles:
            cv2.circle(all_overlay, (x, y), 2, (0, 0, 255), 2)  # Blue dots for final
        
        plt.subplot(1, 6, 5)
        plt.imshow(cv2.cvtColor(all_overlay, cv2.COLOR_BGR2RGB))
        plt.title('All Methods Overlay')
        plt.axis('off')
        
        # Final result
        plt.subplot(1, 6, 6)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Combined Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main(image_path):
    """Main function to demonstrate the bubble photo generation"""
    # Initialize the generator
    generator = BubblePhotoGenerator(
        filled_threshold=0.6,  # Adjust this value to fine-tune filled/unfilled detection
        min_radius=8,          # Minimum bubble radius
        max_radius=25          # Maximum bubble radius
    )


    
    # Input and output paths
    input_image = image_path
    output_image = input_image.replace(".jpg", "_bubble_photo.jpg")
    
    
    try:
        print("Generating combined bubble photo for maximum accuracy...")
        
        # Generate only the combined version for best results
        combined_bubble_photo = generator.generate_combined_bubble_photo(
            input_image,
            output_image,
            show_debug=False  # Set to False to disable debug visualizations
        )
        
        print(f"\nBubble photo generation completed successfully!")
        print(f"Combined result saved as: {output_image}")
        return output_image
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure 'mainsa.jpg' exists in the current directory.")


