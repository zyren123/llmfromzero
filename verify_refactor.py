import torch
import sys
import os
import shutil
from accelerate import Accelerator

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dense import LuluModel, LuluConfig
from models.moe import LuluMoeModel, LuluMoeConfig
from models.vlm import LuluVLModel, LuluVLConfig


def test_model_instantiation():
    print("Testing model instantiation...")

    # Test Lulu
    config = LuluConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2)
    model = LuluModel(config)
    print("LuluModel instantiated successfully.")

    # Test LuluMoe
    config_moe = LuluMoeConfig(
        vocab_size=100, hidden_size=64, num_hidden_layers=2, moe_num_experts=4
    )
    model_moe = LuluMoeModel(config_moe)
    print("LuluMoeModel instantiated successfully.")

    # Test LuluVL
    config_vl = LuluVLConfig(
        text_config={"vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 2}
    )
    model_vl = LuluVLModel(config_vl)
    print("LuluVLModel instantiated successfully.")


def test_auto_model_registration():
    print("\nTesting AutoModel registration...")

    # Create a dummy output directory
    output_dir = "test_lulu_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Save Lulu model
    config = LuluConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2)
    model = LuluModel(config)

    print("Saving LuluModel via save_pretrained...")
    model.save_pretrained(output_dir, safe_serialization=False)

    # Check if code files are copied (this happens if register_for_auto_class worked and we save)
    # Note: save_pretrained only copies code if the config has auto_map and we are saving a model that is registered.
    # Let's check the config.json
    import json

    with open(os.path.join(output_dir, "config.json"), "r") as f:
        saved_config = json.load(f)

    if "auto_map" in saved_config:
        print("Success: auto_map found in config.json")
        print(saved_config["auto_map"])
    else:
        print(
            "Warning: auto_map NOT found in config.json. Registration might not have worked as expected or requires trust_remote_code=True flow verification."
        )

    # Clean up
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def test_accelerate_import():
    print("\nTesting Accelerator import...")
    accelerator = Accelerator()
    print(f"Accelerator initialized on {accelerator.device}")


if __name__ == "__main__":
    test_model_instantiation()
    test_auto_model_registration()
    test_accelerate_import()
