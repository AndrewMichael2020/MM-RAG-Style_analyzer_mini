# Style Finder - Multimodal RAG Fashion Analyzer

**Developed by: Hailey Quach, IBM Instructor**  
**Implementer: Andrew Ihnativ**  
**Course lab: Build a Style Finder Using Multimodal Retrieval and Search**  
**Organization: IBM Skills Network**

---

## Overview

Style Finder is a powerful multimodal AI application that combines computer vision, vector similarity search, and large language models to analyze fashion items from images. Built as part of IBM's educational curriculum, this application demonstrates the power of combining multimodal AI with Retrieval-Augmented Generation (RAG) to create an educational and practical fashion analysis tool.

The system identifies clothing items, analyzes their style characteristics, and provides users with detailed information including brand details and pricing suggestions through an intuitive web interface.

## Features

- **🔍 Image Analysis**: Upload fashion images and get instant AI-powered style analysis
- **👗 Style Recognition**: Identifies clothing items, colors, patterns, and style characteristics
- **🎯 Similarity Matching**: Finds visually similar items using advanced vector embeddings
- **🤖 AI-Powered Descriptions**: Generates detailed, professional fashion descriptions using Llama 3.2 Vision Instruct
- **🛒 Shopping Integration**: Searches for similar items across the web with pricing information
- **📱 Interactive Web Interface**: User-friendly Gradio interface with example images

## Architecture

The application uses a sophisticated multimodal RAG pipeline:

1. **Image Processing**: ResNet50-based feature extraction and encoding
2. **Vector Similarity**: Cosine similarity matching against a pre-computed fashion database
3. **LLM Analysis**: Llama 3.2 Vision Instruct model for detailed fashion analysis
4. **Search Integration**: Optional SerpAPI integration for real-time product searches

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- Access to IBM Watson AI services
- SerpAPI key (optional, for shopping features)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MM-RAG-Style_analyzer_mini
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   # Set up IBM Watson AI credentials
   export WATSONX_APIKEY=your_api_key
   export WATSONX_URL=your_watson_url
   
   # Optional: Set SerpAPI key for shopping features
   export SERP_API_KEY=your_serp_api_key
   ```

4. **Download the dataset**:
   Ensure you have the `swift-style-embeddings.pkl` file containing the fashion database with pre-computed embeddings.

## Usage

### Quick Start

1. **Launch the application**:
   ```bash
   python app.py
   ```

2. **Access the interface**:
   Open your browser and navigate to `http://127.0.0.1:5000`

3. **Analyze fashion images**:
   - Upload an image or use the provided examples
   - Adjust settings for alternative suggestions
   - Click "Analyze Style" to get AI-powered insights

### Configuration

The application can be customized through `config.py`:

```python
# Model configuration
LLAMA_MODEL_ID = "meta-llama/llama-3-2-90b-vision-instruct"
PROJECT_ID = "skills-network"
REGION = "us-south"

# Image processing settings
IMAGE_SIZE = (224, 224)
SIMILARITY_THRESHOLD = 0.8
DEFAULT_ALTERNATIVES_COUNT = 5
```

## Project Structure

```
MM-RAG-Style_analyzer_mini/
├── app.py                     # Main Gradio application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── models/
│   ├── image_processor.py     # Image encoding and similarity matching
│   └── llm_service.py         # Llama Vision service integration
├── services/
│   └── search_service.py      # SerpAPI integration for product search
├── utils/
│   └── helpers.py             # Utility functions
└── examples/
    ├── test-6.png             # Sample fashion images
    └── test-7.png
```

## Key Components

### ImageProcessor
- **Feature Extraction**: Uses pre-trained ResNet50 for image encoding
- **Similarity Matching**: Implements cosine similarity for finding closest matches
- **Preprocessing**: Handles image normalization and resizing

### LlamaVisionService
- **AI Analysis**: Leverages Llama 3.2 Vision Instruct for fashion insights
- **Contextual Understanding**: Combines visual analysis with database information
- **Professional Descriptions**: Generates detailed style and brand information

### SearchService
- **Product Discovery**: Integrates with SerpAPI for real-time product searches
- **Price Comparison**: Finds similar items with pricing information
- **Web Integration**: Expands beyond the local database for comprehensive results

## Example Usage

```python
from app import StyleFinderApp

# Initialize the application
app = StyleFinderApp("swift-style-embeddings.pkl", serp_api_key="your_key")

# Process an image
result = app.process_image(
    image="path/to/fashion/image.jpg",
    alternatives_count=5,
    include_alternatives=True
)

print(result)
```

## API Integration

The application integrates with several services:

- **IBM Watson AI**: For Llama 3.2 Vision model access
- **SerpAPI** (optional): For real-time product searches
- **Gradio**: For the web interface

## Extending the Application

### Adding New Fashion Items
1. Process new images through the ImageProcessor
2. Generate embeddings using the same ResNet50 model
3. Update the dataset pickle file with new entries

### Custom Vision Models
```python
# Replace ResNet50 with your preferred model
from torchvision.models import efficientnet_b0
self.model = efficientnet_b0(pretrained=True)
```

### Enhanced Search Integration
- Add support for additional search APIs
- Implement price tracking and alerts
- Integrate with e-commerce platforms

## Performance Optimization

- **GPU Acceleration**: Utilizes CUDA when available for faster image processing
- **Batch Processing**: Supports processing multiple images efficiently
- **Caching**: Implements smart caching for repeated queries
- **Model Optimization**: Uses evaluation mode for inference-only operations

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   ```bash
   # Ensure proper Watson AI credentials
   export WATSONX_APIKEY=your_actual_api_key
   ```

2. **Image Processing Failures**:
   - Check image format (supported: JPG, PNG, JPEG)
   - Verify image file is not corrupted
   - Ensure sufficient memory for large images

3. **Similarity Search Issues**:
   - Verify dataset file exists and is accessible
   - Check for proper pickle file format
   - Ensure embeddings are compatible

## Development and Testing

### Running Tests
```bash
# Test image processing
python -m pytest tests/test_image_processor.py

# Test LLM integration
python -m pytest tests/test_llm_service.py
```

### Development Mode
```bash
# Run with debug mode
python app.py --debug
```

## Future Enhancements

Potential areas for expansion:
- **Real-time API Integration**: Connect to live fashion databases
- **Advanced Filtering**: Add size, color, and price filters
- **User Personalization**: Implement user preference learning
- **Mobile App**: Develop native mobile applications
- **Social Features**: Add sharing and community features
- **AR Integration**: Implement augmented reality try-on features

## Contributing

This project was developed as part of IBM's educational curriculum. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Author

**Hailey Quach** - IBM Instructor

## License

This project is part of IBM's educational curriculum and is intended for learning purposes.

## Acknowledgments

- IBM Watson AI team for providing the AI services
- Meta for the Llama 3.2 Vision Instruct model
- The fashion dataset contributors
- Gradio team for the excellent interface framework

## Support

For support and questions related to this educational project:
- Check the troubleshooting section above
- Review the IBM skills network documentation
- Contact your course instructor

---

*Built with ❤️ using IBM Watson AI, Llama 3.2 Vision, and modern Python frameworks*
