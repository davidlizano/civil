import os
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_comparison.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ImageComparator:
    def __init__(self, image_a_path, image_b_path):
        self.image_a_path = image_a_path
        self.image_b_path = image_b_path
        self.image_a = None
        self.image_b = None
        self.differences = []
        
    def load_images(self):
        """Load and validate images"""
        try:
            self.image_a = cv2.imread(self.image_a_path)
            self.image_b = cv2.imread(self.image_b_path)
            
            if self.image_a is None or self.image_b is None:
                raise ValueError("Could not load one or both images")
                
            # Ensure images have the same dimensions
            if self.image_a.shape != self.image_b.shape:
                self.image_b = cv2.resize(self.image_b, (self.image_a.shape[1], self.image_a.shape[0]))
                
            logging.info(f"Images loaded successfully. Shape: {self.image_a.shape}")
            return True
        except Exception as e:
            logging.error(f"Error loading images: {str(e)}")
            return False

    def find_differences(self):
        """Find differences between images using multiple methods"""
        try:
            # Convert images to grayscale
            gray_a = cv2.cvtColor(self.image_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(self.image_b, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_a, gray_b)
            
            # Apply threshold to get significant differences
            _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up the difference mask
            kernel = np.ones((7,7), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours of differences
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and process contours
            min_contour_area = 1000
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Expand the region by 20% in each direction for better context
                    context_padding = int(max(w, h) * 0.2)
                    x = max(0, x - context_padding)
                    y = max(0, y - context_padding)
                    w += 2 * context_padding
                    h += 2 * context_padding
                    
                    # Ensure we don't go beyond image boundaries
                    x = min(x, self.image_a.shape[1] - w)
                    y = min(y, self.image_a.shape[0] - h)
                    
                    # Extract regions from both images
                    region_a = self.image_a[y:y+h, x:x+w]
                    region_b = self.image_b[y:y+h, x:x+w]
                    
                    # Save regions temporarily for analysis
                    temp_a_path = f"temp_region_a_{x}_{y}.jpg"
                    temp_b_path = f"temp_region_b_{x}_{y}.jpg"
                    cv2.imwrite(temp_a_path, region_a)
                    cv2.imwrite(temp_b_path, region_b)
                    
                    # Analyze the difference using OpenAI
                    analysis = self._analyze_difference(region_a, region_b, (x, y, w, h))
                    if analysis:
                        self.differences.append(analysis)
                    
                    # Clean up temporary files
                    os.remove(temp_a_path)
                    os.remove(temp_b_path)
            
            logging.info(f"Found {len(self.differences)} significant differences")
            return True
        except Exception as e:
            logging.error(f"Error finding differences: {str(e)}")
            return False

    def _analyze_difference(self, region_a, region_b, bbox):
        """Analyze a difference region using OpenAI Vision API"""
        try:
            # Convert regions to base64
            _, buffer_a = cv2.imencode('.jpg', region_a)
            _, buffer_b = cv2.imencode('.jpg', region_b)
            base64_a = base64.b64encode(buffer_a).decode('utf-8')
            base64_b = base64.b64encode(buffer_b).decode('utf-8')
            
            prompt = """
            Compare these two image regions taken at different times. The regions include extra context around the potential changes.
            
            For each apparent change, please analyze:
            1. Is this a real change (element added/removed) or just appears different due to:
               - Perspective/viewing angle
               - Lighting conditions
               - Partial visibility
               - Camera position
            2. If it's a real change, what elements were:
               - Added (present in second image but not in first)
               - Removed (present in first image but not in second)
            3. If it's not a real change, explain why it appears different

            Format your response as a JSON object with these keys:
            - is_real_change: boolean indicating if this is a real change or just perspective/lighting
            - change_type: "added", "removed", or "perspective" if not a real change
            - elements: list of elements that changed (if real change)
            - explanation: explanation of why it appears different (if not real change)
            - confidence: your confidence in the analysis (high/medium/low)
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_a}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_b}"}}
                        ]
                    }
                ],
                response_format={ "type": "json_object" },
                max_tokens=800
            )
            
            analysis = json.loads(response.choices[0].message.content)
            analysis['bbox'] = bbox
            return analysis
        except Exception as e:
            logging.error(f"Error analyzing difference: {str(e)}")
            return None

    def visualize_differences(self):
        """Create visualization of the differences"""
        try:
            # Create a copy of the first image for visualization
            vis_image = self.image_a.copy()
            
            # Define colors for real changes only
            added_color = (0, 255, 0)     # Green for added elements
            removed_color = (0, 0, 255)    # Red for removed elements
            
            # Draw rectangles and labels only for real changes
            for diff in self.differences:
                # Skip if it's not a real change
                if not diff['is_real_change']:
                    continue
                    
                x, y, w, h = diff['bbox']
                
                # Determine color and label based on change type
                if diff['change_type'] == 'added':
                    color = added_color
                    label = f"Added: {', '.join(diff['elements'])}"
                else:  # removed
                    color = removed_color
                    label = f"Removed: {', '.join(diff['elements'])}"
                
                # Draw rectangle
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis_image, (x, y - text_height - 10), (x + text_width, y), color, -1)
                
                # Draw label text
                cv2.putText(vis_image, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add confidence level
                conf_label = f"Confidence: {diff['confidence']}"
                cv2.putText(vis_image, conf_label, (x, y + h + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save the visualization
            vis_path = 'difference_visualization.jpg'
            cv2.imwrite(vis_path, vis_image)
            return vis_path
        except Exception as e:
            logging.error(f"Error creating visualization: {str(e)}")
            return None

    def save_results(self):
        """Save the analysis results and visualization"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "comparison_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save the visualization
            vis_path = self.visualize_differences()
            if vis_path:
                os.rename(vis_path, f"{results_dir}/differences_{timestamp}.jpg")
            
            # Save the analysis text
            with open(f"{results_dir}/analysis_{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write("Image Comparison Analysis\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                # Count real changes
                real_changes = [d for d in self.differences if d['is_real_change']]
                f.write(f"Found {len(real_changes)} real changes (ignoring perspective/lighting changes):\n\n")
                
                for i, diff in enumerate(real_changes, 1):
                    f.write(f"Change #{i}\n")
                    f.write(f"Location: {diff['bbox']}\n")
                    f.write(f"Change Type: {diff['change_type']}\n")
                    f.write(f"Elements: {', '.join(diff['elements'])}\n")
                    f.write(f"Confidence: {diff['confidence']}\n")
                    f.write("\n")
                
                # Add a section for perspective changes if any
                perspective_changes = [d for d in self.differences if not d['is_real_change']]
                if perspective_changes:
                    f.write("\nPerspective/Lighting Changes (not shown in visualization):\n")
                    for i, diff in enumerate(perspective_changes, 1):
                        f.write(f"\nApparent Change #{i}\n")
                        f.write(f"Location: {diff['bbox']}\n")
                        f.write(f"Explanation: {diff['explanation']}\n")
                        f.write(f"Confidence: {diff['confidence']}\n")
            
            logging.info(f"Results saved in {results_dir}")
            return True
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            return False

def main():
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logging.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # Initialize comparator
    comparator = ImageComparator('fotoA.jpg', 'fotoB.jpg')
    
    # Process images
    if not comparator.load_images():
        return
    
    # Find differences
    if not comparator.find_differences():
        return
    
    # Save results
    comparator.save_results()
    
    # Print summary
    real_changes = [d for d in comparator.differences if d['is_real_change']]
    print("\nAnalysis Results:")
    print(f"Found {len(real_changes)} real changes (ignoring perspective/lighting changes)")
    for i, diff in enumerate(real_changes, 1):
        print(f"\nChange #{i}")
        print(f"Location: {diff['bbox']}")
        print(f"Change Type: {diff['change_type']}")
        print(f"Elements: {', '.join(diff['elements'])}")
        print(f"Confidence: {diff['confidence']}")
    
    # Print perspective changes if any
    perspective_changes = [d for d in comparator.differences if not d['is_real_change']]
    if perspective_changes:
        print("\nPerspective/Lighting Changes (not shown in visualization):")
        for i, diff in enumerate(perspective_changes, 1):
            print(f"\nApparent Change #{i}")
            print(f"Location: {diff['bbox']}")
            print(f"Explanation: {diff['explanation']}")
            print(f"Confidence: {diff['confidence']}")

if __name__ == "__main__":
    main()
