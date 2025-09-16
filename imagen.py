import os
import json
import base64
import requests
import argparse
from datetime import datetime

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

BASE_URL = "https://api.openai.com/v1/images"
GPT_IMAGE_MODEL = "gpt-image-1"

# --- Helper Functions ---
def save_image_from_b64(b64_string, prefix, index, output_format="png"):
    """Decodes b64 string and saves image."""
    try:
        image_data = base64.b64decode(b64_string)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{index+1}.{output_format}"
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"Saved image: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def handle_api_response(response, output_prefix, expected_output_format_for_saving="png"):
    """Processes API response and saves images."""
    if response.status_code == 200:
        try:
            data = response.json()
            # print(f"API Response JSON: {json.dumps(data, ensure_ascii=False, indent=2)}") # Debugging

            if "data" in data and isinstance(data["data"], list):
                for i, item in enumerate(data["data"]):
                    if "b64_json" in item:
                        save_image_from_b64(item["b64_json"], output_prefix, i, expected_output_format_for_saving)
                    else:
                        print(f"Warning: Item {i} does not contain \"b64_json\". Item: {item}")
            else:
                print(f"Unexpected response structure: \"data\" field missing or not a list. Full response: {data}")

            if "usage" in data and data["usage"] is not None: # gpt-image-1 provides usage
                 print(f"\nUsage Information:")
                 print(f"  Total Tokens: {data["usage"].get("total_tokens")}")
                 print(f"  Input Tokens: {data["usage"].get("input_tokens")}")
                 print(f"  Output Tokens: {data["usage"].get("output_tokens")}")
                 if "input_tokens_details" in data["usage"]:
                     print(f"  Input Text Tokens: {data["usage"]["input_tokens_details"].get("text_tokens")}")
                     print(f"  Input Image Tokens: {data["usage"]["input_tokens_details"].get("image_tokens")}")
            elif "data" in data and not data["data"]: # Check if data is empty list
                 print("No images were generated. This might be due to a safety filter or other issue.")
            elif not ("data" in data and isinstance(data["data"], list)):
                print("Warning: No usage information provided in the response, and data structure is unexpected.")

        except json.JSONDecodeError:
            print("Error: Could not decode JSON response.")
            print(f"Raw response: {response.text}")
        except Exception as e:
            print(f"An error occurred while processing the response: {e}")
            print(f"Raw response JSON: {response.json()}")

    else:
        print(f"Error: API request failed with status code {response.status_code}")
        try:
            error_details = response.json()
            print(f"Error details: {json.dumps(error_details, ensure_ascii=False, indent=2)}")
        except json.JSONDecodeError:
            print(f"Could not parse error response: {response.text}")

# --- API Call Functions ---
def create_image(args):
    """Handles the "create" image command."""
    print(f"Creating image with prompt: \"{args.prompt}\" using model {GPT_IMAGE_MODEL}")
    url = f"{BASE_URL}/generations"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": args.prompt,
        "model": GPT_IMAGE_MODEL,
        "n": args.n,
        "quality": args.quality,
        "size": args.size,
        "background": args.background,
        "output_format": args.output_format # This is for gpt-image-1
        # response_format is not used for gpt-image-1 as it always returns b64_json
    }
    if args.user:
        payload["user"] = args.user
    if args.output_compression is not None and args.output_format in ["jpeg", "webp"]:
        payload["output_compression"] = args.output_compression
    if args.moderation:
        payload["moderation"] = args.moderation

    # print(f"Sending payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120) # Increased timeout
        handle_api_response(response, args.output_prefix, args.output_format)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def edit_image(args):
    """Handles the "edit" image command."""
    image_paths_str = ", ".join([f"\"{path}\"" for path in args.image_paths])
    print(f"Editing image(s) {image_paths_str} with prompt: \"{args.prompt}\" using model {GPT_IMAGE_MODEL}")
    if len(args.image_paths) > 16:
        print(f"Warning: You provided {len(args.image_paths)} images. gpt-image-1 supports up to 16 images for edits. Proceeding, but the API might reject this.")
    if args.mask_path and len(args.image_paths) > 1:
        print(f"Note: A mask was provided with multiple images. The mask will be applied to the first image: \"{args.image_paths[0]}\".")

    url = f"{BASE_URL}/edits"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
        # Content-Type will be multipart/form-data, set by requests
    }

    files_to_send = []
    opened_files = [] # To keep track of file objects for closing

    try:
        for image_path in args.image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            file_obj = open(image_path, "rb")
            opened_files.append(file_obj)
            files_to_send.append(("image[]", (os.path.basename(image_path), file_obj)))

        if args.mask_path:
            if not os.path.exists(args.mask_path):
                raise FileNotFoundError(f"Mask file not found: {args.mask_path}")
            mask_file_obj = open(args.mask_path, "rb")
            opened_files.append(mask_file_obj)
            files_to_send.append(("mask", (os.path.basename(args.mask_path), mask_file_obj)))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        for f_obj in opened_files: # Close any files already opened
            f_obj.close()
        return
    except Exception as e:
        print(f"Error opening file(s): {e}")
        for f_obj in opened_files: # Close any files already opened
            f_obj.close()
        return

    data_payload = {
        "prompt": args.prompt,
        "model": GPT_IMAGE_MODEL,
        "n": args.n,
        "quality": args.quality,
        "size": args.size
    }
    if args.user:
        data_payload["user"] = args.user

    # print(f"Sending data payload: {json.dumps(data_payload, ensure_ascii=False, indent=2)}")
    # print(f"Sending files: {files_to_send}") # For debugging the structure of files_to_send

    try:
        response = requests.post(url, headers=headers, files=files_to_send, data=data_payload, timeout=120) # Increased timeout
        handle_api_response(response, args.output_prefix, "png")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    finally:
        for f_obj in opened_files:
            f_obj.close()

# --- Main Argument Parser ---
def main():
    parser = argparse.ArgumentParser(description="OpenAI Image CLI for gpt-image-1 model.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command help")

    # --- Create Sub-parser ---
    create_parser = subparsers.add_parser("create", help="Create an image from a text prompt.")
    create_parser.add_argument("prompt", type=str, help="A text description of the desired image(s). Max 32000 chars for gpt-image-1.")
    create_parser.add_argument("--n", type=int, default=1, choices=range(1, 11), help="Number of images to generate (1-10).")
    create_parser.add_argument("--quality", type=str, default="low", choices=["low", "medium", "high", "auto"], help="Quality of the image for gpt-image-1.")
    create_parser.add_argument("--size", type=str, default="1024x1024", choices=["auto", "1024x1024", "1536x1024", "1024x1536"], help="Size of the generated images for gpt-image-1.")
    create_parser.add_argument("--background", type=str, default="opaque", choices=["transparent", "opaque", "auto"], help="Background mode for gpt-image-1 (e.g., \"transparent\").")
    create_parser.add_argument("--output-format", type=str, default="png", choices=["png", "jpeg", "webp"], help="Output format for generated images (gpt-image-1 only).")
    create_parser.add_argument("--output-compression", type=int, default=None, choices=range(0,101), metavar="[0-100]", help="Compression level (0-100) for \"jpeg\" or \"webp\" output (gpt-image-1 only). Defaults to 100.")
    create_parser.add_argument("--moderation", type=str, default="low", choices=["low", "auto"], help="Content moderation level for gpt-image-1.")
    create_parser.add_argument("--user", type=str, help="A unique identifier for your end-user.")
    create_parser.add_argument("--output-prefix", type=str, default="gen", help="Prefix for the output image filename(s).")
    create_parser.set_defaults(func=create_image)

    # --- Edit Sub-parser ---
    edit_parser = subparsers.add_parser("edit", help="Edit existing image(s) based on a new prompt.")
    edit_parser.add_argument("image_paths", type=str, nargs='+', help="Path(s) to the source image(s) (png, webp, or jpg, <25MB each for gpt-image-1). Provide up to 16 images.")
    edit_parser.add_argument("prompt", type=str, help="A text description of the desired edit. Max 32000 chars for gpt-image-1.")
    edit_parser.add_argument("--mask-path", type=str, help="Optional: Path to a mask image (PNG, same dimensions as source). If multiple images are provided, mask applies to the first image.")
    edit_parser.add_argument("--n", type=int, default=1, choices=range(1, 11), help="Number of edited images to generate (1-10).")
    edit_parser.add_argument("--quality", type=str, default="low", choices=["low", "medium", "high", "auto"], help="Quality of the image for gpt-image-1.")
    edit_parser.add_argument("--size", type=str, default="1024x1024", choices=["auto", "1024x1024", "1536x1024", "1024x1536"], help="Size of the generated images for gpt-image-1. \"auto\" uses original image size if possible.")
    edit_parser.add_argument("--user", type=str, help="A unique identifier for your end-user.")
    edit_parser.add_argument("--output-prefix", type=str, default="edi", help="Prefix for the output image filename(s).")
    edit_parser.set_defaults(func=edit_image)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
