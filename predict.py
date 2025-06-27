# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random
import tempfile
import yaml
import logging
import traceback
import torch
import numpy as np
from PIL import Image

from cog import BasePredictor, Input, Path as CogPath

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Using device: {self.device}")

            # Validate GPU availability for production
            if self.device == "cuda":
                gpu_count = torch.cuda.device_count()
                gpu_memory = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )
                logger.info(f"GPU Info: {gpu_count} GPU(s), {gpu_memory:.1f}GB memory")

                if gpu_memory < 8:
                    logger.warning(
                        "⚠️ Less than 8GB GPU memory detected. Consider using 2B model."
                    )

            # Set environment variables
            os.environ["MODEL_DIR"] = "./models"
            os.environ["HF_HOME"] = "./models"

            # Find and validate configuration
            self.config_path = self._find_valid_config()
            logger.info(f"📋 Using config: {self.config_path}")

            # Validate config file
            self._validate_config()

            # Import and create pipeline
            self._import_ltx_modules()
            self.pipeline = self._create_pipeline()

            logger.info("✅ LTX-Video model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _find_valid_config(self) -> str:
        """Find a valid configuration file with proper fallback"""
        possible_configs = [
            "configs/ltxv-2b-0.9.6-distilled.yaml",  # Start with 2B for better compatibility
            "configs/ltxv-13b-0.9.7-distilled.yaml",
            "configs/ltxv-2b-0.9.6-dev.yaml",
            "configs/ltxv-13b-0.9.7-dev.yaml",
        ]

        for config in possible_configs:
            if os.path.exists(config):
                return config

        raise FileNotFoundError(
            f"No valid configuration file found. Please ensure one of these exists: {possible_configs}"
        )

    def _validate_config(self) -> None:
        """Validate the configuration file"""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            # Check required fields
            required_fields = ["checkpoint_path"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in config")

            # Validate model path
            model_path = config["checkpoint_path"]
            if not os.path.exists(model_path):
                # Try alternative paths
                alternative_paths = [
                    f"models/{os.path.basename(model_path)}",
                    f"./models/{os.path.basename(model_path)}",
                ]

                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        logger.info(f"Found model at alternative path: {alt_path}")
                        # Update config in memory (not on disk)
                        config["checkpoint_path"] = alt_path
                        self._config = config
                        return

                raise FileNotFoundError(
                    f"Model not found at {model_path} or alternative paths"
                )

            self._config = config

        except Exception as e:
            raise ValueError(f"Invalid configuration file {self.config_path}: {str(e)}")

    def _import_ltx_modules(self) -> None:
        """Import LTX-Video modules with error handling"""
        try:
            # These imports might fail if the module structure is different
            global create_ltx_video_pipeline, infer
            from inference import create_ltx_video_pipeline, infer
        except ImportError as e:
            logger.error(f"Failed to import LTX-Video modules: {e}")
            raise ImportError(
                "Cannot import LTX-Video modules. Ensure the repository structure is correct "
                "and all dependencies are installed."
            )

    def _create_pipeline(self):
        """Create the LTX-Video pipeline"""
        try:
            model_path = self._config["checkpoint_path"]

            pipeline = create_ltx_video_pipeline(
                ckpt_path=model_path,
                precision="fp16" if self.device == "cuda" else "fp32",
                text_encoder_model_name_or_path="google/flan-t5-xl",
                device=self.device,
                enhance_prompt=True,
            )

            return pipeline

        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            raise

    def _validate_dimensions(self, width: int, height: int, num_frames: int) -> tuple:
        """Validate and potentially adjust dimensions"""
        # Ensure divisibility by 32
        if width % 32 != 0:
            width = (width // 32) * 32
            logger.warning(f"Width adjusted to {width} (must be divisible by 32)")

        if height % 32 != 0:
            height = (height // 32) * 32
            logger.warning(f"Height adjusted to {height} (must be divisible by 32)")

        # Ensure frames follow 8*n + 1 pattern
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
            logger.warning(f"Frames adjusted to {num_frames} (must be 8*n + 1)")

        # Check recommended limits
        if width * height > 720 * 1280:
            logger.warning("Resolution exceeds recommended limits, may cause OOM")

        if num_frames > 257:
            logger.warning("Frame count exceeds recommended limits")

        return width, height, num_frames

    def _create_temp_image(self, image_path: str) -> str:
        """Create a temporary image file with proper cleanup tracking"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                pil_image = Image.open(image_path).convert("RGB")
                pil_image.save(tmp_file.name, "PNG")
                return tmp_file.name
        except Exception as e:
            logger.error(f"Failed to process input image: {str(e)}")
            raise ValueError(f"Invalid input image: {str(e)}")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the video you want to generate",
            default="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek.",
        ),
        image: CogPath = Input(
            description="Input image for image-to-video generation (optional)",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Negative prompt to avoid certain content",
            default="worst quality, inconsistent motion, blurry, jittery, distorted, watermarks",
        ),
        width: int = Input(
            description="Width of the output video (must be divisible by 32)",
            default=768,
            choices=[512, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216],
        ),
        height: int = Input(
            description="Height of the output video (must be divisible by 32)",
            default=512,
            choices=[512, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216],
        ),
        num_frames: int = Input(
            description="Number of frames to generate (must be 8*n + 1, e.g., 25, 49, 121)",
            default=121,
            choices=[25, 49, 73, 97, 121, 145, 169, 193, 217, 241],
        ),
        frame_rate: int = Input(
            description="Frame rate for the output video",
            default=30,
            choices=[24, 25, 30],
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance (higher values = more prompt adherence)",
            default=3.0,
            ge=1.0,
            le=10.0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (more steps = higher quality, slower generation)",
            default=30,
            ge=8,
            le=50,
        ),
        seed: int = Input(
            description="Random seed for reproducible results. Leave blank to randomize",
            default=None,
        ),
        conditioning_start_frame: int = Input(
            description="Frame index to start conditioning from (only used with input image)",
            default=0,
            ge=0,
        ),
        conditioning_strength: float = Input(
            description="Strength of conditioning when using input image",
            default=1.0,
            ge=0.0,
            le=2.0,
        ),
    ) -> CogPath:
        """Run a single prediction on the model"""

        temp_files = []  # Track temporary files for cleanup

        try:
            # Input validation
            if not prompt or len(prompt.strip()) == 0:
                raise ValueError("Prompt cannot be empty")

            if len(prompt) > 1000:
                logger.warning("Prompt is very long, truncating to 1000 characters")
                prompt = prompt[:1000]

            # Validate and adjust dimensions
            width, height, num_frames = self._validate_dimensions(
                width, height, num_frames
            )

            # Set random seed
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            elif seed < 0 or seed > 2**32 - 1:
                seed = abs(seed) % (2**32)

            # Set all random seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed % (2**31))  # numpy wants smaller range
            random.seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            logger.info(f"🎲 Using seed: {seed}")
            logger.info(
                f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
            )
            logger.info(
                f"📐 Dimensions: {width}x{height}, {num_frames} frames @ {frame_rate}fps"
            )

            # Prepare conditioning for image-to-video
            conditioning_media_paths = []
            conditioning_start_frames = []
            conditioning_strengths = []

            if image is not None:
                logger.info("🖼️ Using image-to-video mode")
                temp_image_path = self._create_temp_image(str(image))
                temp_files.append(temp_image_path)

                conditioning_media_paths.append(temp_image_path)
                conditioning_start_frames.append(conditioning_start_frame)
                conditioning_strengths.append(conditioning_strength)
            else:
                logger.info("📝 Using text-to-video mode")

            # Create output path
            output_path = f"/tmp/ltx_video_{seed}.mp4"

            # Generate video using the original inference function
            logger.info("🎬 Starting video generation...")

            # Call the original inference function with proper error handling
            infer(
                output_path=output_path,
                seed=seed,
                pipeline_config=self.config_path,
                image_cond_noise_scale=0.02,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                prompt=prompt,
                negative_prompt=negative_prompt,
                offload_to_cpu=False,
                conditioning_media_paths=(
                    conditioning_media_paths if conditioning_media_paths else None
                ),
                conditioning_strengths=(
                    conditioning_strengths if conditioning_strengths else None
                ),
                conditioning_start_frames=(
                    conditioning_start_frames if conditioning_start_frames else None
                ),
                device=self.device,
            )

            # Validate output file
            if not os.path.exists(output_path):
                raise RuntimeError(
                    "Video generation completed but output file was not created"
                )

            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise RuntimeError("Generated video file is empty")

            logger.info(
                f"✅ Video generation completed: {output_path} ({file_size/1024/1024:.1f}MB)"
            )
            return CogPath(output_path)

        except Exception as e:
            logger.error(f"❌ Error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
