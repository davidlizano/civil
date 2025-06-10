# Image Comparison Tool

This tool compares two images taken from similar positions at different times and uses OpenAI's Vision API to analyze the differences between them. It can identify new objects, changes in lighting, structural changes, and provide a detailed analysis of the differences.

## Features

- Image preprocessing for better comparison
- Structural Similarity Index (SSIM) for difference detection
- OpenAI Vision API integration for detailed analysis
- Automatic visualization of differences
- Comprehensive logging
- Results saving with timestamps

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Place your images in the project directory:
   - `fotoA.jpg` (original image)
   - `fotoB.jpg` (comparison image)

2. Run the script:
   ```bash
   python main.py
   ```

3. The script will:
   - Load and preprocess the images
   - Find differences using SSIM
   - Analyze the differences using OpenAI Vision API
   - Save results in the `comparison_results` directory
   - Display the analysis in the console

## Output

The script generates:
- A visualization of the differences (saved as `differences_TIMESTAMP.jpg`)
- A text file with the analysis (saved as `analysis_TIMESTAMP.txt`)
- A log file (`image_comparison.log`) with detailed execution information

## Notes

- Both images should be in JPG format
- Images should be taken from similar positions for best results
- The script will automatically resize images if they have different dimensions
- Make sure you have sufficient OpenAI API credits for the analysis

## Error Handling

The script includes comprehensive error handling and logging. Check the `image_comparison.log` file for detailed information about any issues that occur during execution. 