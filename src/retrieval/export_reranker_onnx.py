import os
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "BAAI/bge-reranker-base"
EXPORT_DIR = Path("models/onnx_reranker_quantized")
RAW_ONNX_PATH = EXPORT_DIR / "model.onnx"
QUANT_ONNX_PATH = EXPORT_DIR / "model_quantized.onnx"

def export_and_quantize():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    # Wrap model to ensure clean single-tensor output for ONNX tracer
    class RerankerWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    wrapped_model = RerankerWrapper(model)
    wrapped_model.eval()

    dummy_text_pair = [("cancer inquiry", "histopathological tissue sample")]
    dummy_inputs = tokenizer(
        dummy_text_pair,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )

    print(f"Exporting TorchScript ONNX graph -> {RAW_ONNX_PATH}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            f=str(RAW_ONNX_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "num_labels"}
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False  # Disables torch._dynamo, uses stable TorchScript exporter
        )

    tokenizer.save_pretrained(EXPORT_DIR)
    print("Base ONNX export completed.")

    print(f"Applying dynamic INT8 quantization -> {QUANT_ONNX_PATH}...")
    quantize_dynamic(
        model_input=str(RAW_ONNX_PATH),
        model_output=str(QUANT_ONNX_PATH),
        op_types_to_quantize=["MatMul", "Gemm"],  # Quantize transformer matrix multiplications
        weight_type=QuantType.QInt8
    )

    if RAW_ONNX_PATH.exists():
        RAW_ONNX_PATH.unlink()

    print(f"\nINT8 ONNX reranker successfully created at: {QUANT_ONNX_PATH}")

if __name__ == "__main__":
    export_and_quantize()