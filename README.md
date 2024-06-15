# ADetailer for Stable Diffusion XL

This project is a modified version of [ADetailer](https://github.com/Bing-su/adetailer), an extension for the A1111 WebUI for Stable Diffusion. This modification allows the package to run directly through code, enabling users to import it, follow the provided example Python file, and perform inference. It can also be integrated into other projects.

## Features

- **Integration with Stable Diffusion XL**: This modified version works specifically with Stable Diffusion XL, a text-to-image model.
- **Easy Inference**: Users can easily make inferences by following the example provided.
- **Seamless Integration**: This package can be integrated into other projects without the need for the A1111 WebUI.

## Sample Inference

1. Clone the repository:
    ```sh
    git clone https://github.com/theblackhatmagician/adetailer_sdxl.git
    cd adetailer_sdxl
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use this package, import it in your Python script and follow the example script provided.


![Original Generated Image:](image.png)

![Image after ADetailer Processing:](image_fix.png)

## Acknowledgments
- Bing-su for the original ADetailer.
- The A1111 WebUI team for their work on the WebUI for Stable Diffusion.