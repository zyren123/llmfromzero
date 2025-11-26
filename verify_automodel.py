import os
import shutil
from transformers import AutoConfig, AutoModelForCausalLM
from models.dense import LuluConfig, LuluModel

# 1. Register the custom classes
# This tells AutoConfig that "dense_qwen" model_type corresponds to DenseConfig
AutoConfig.register("lulu", LuluConfig)
# This tells AutoModel that when it sees DenseConfig, it should use DenseModel
AutoModelForCausalLM.register(LuluConfig, LuluModel)


def test_save_and_load():
    save_directory = "tmp_test_model"

    # Clean up if exists
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)

    print("1. Initializing a small random model...")
    config = LuluConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LuluModel(config)

    print(f"2. Saving model to {save_directory}...")
    model.save_pretrained(save_directory, safe_serialization=False)
    # Also save tokenizer if we had one, but here we focus on model

    print("3. Loading model using AutoModelForCausalLM...")
    try:
        loaded_model = AutoModelForCausalLM.from_pretrained(save_directory)
        print("   Success! Model loaded.")
        print(f"   Loaded model class: {type(loaded_model)}")

        # Verify it's indeed our class
        assert isinstance(loaded_model, LuluModel)
        print("   Verification passed: Loaded model is instance of LuluModel.")

    except Exception as e:
        print(f"   Failed to load: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory)


if __name__ == "__main__":
    test_save_and_load()
