# Benchmark Configs

Each file answers three simple questions:

- `training_fixed`: which parameters are fixed during training
- `inference_fixed`: which parameters are fixed during inference
- `validation_selected`: which parameters are chosen by validation and then reused later

Optional:

- `validation_fixed`: fixed settings used to run the validation procedure itself

You can also put optional path overrides directly in the config instead of the CLI:

- `training_fixed`: for keys such as `alpaca_json`, `behavior_refusal_json`, `condition_json`, `vector_out`, `condition_vector_out`, `refusal_analysis_out_dir`, `condition_analysis_out_dir`, or `force_refusal`
- `validation_fixed`: for keys such as `calibration_json`, `condition_point_out`, or `condition_point_analysis_out`

This optional section exists because some methods, like CAST, do not only have validation outputs such as a chosen threshold. They also have fixed search settings such as a threshold range or layer range.

Recommended `validation_selected` pattern:

- Use `default` for the fallback value used when no validation report exists.
- Use `values` for the search grid explored during validation.

Example:

```json
"validation_selected": {
  "strength": {
    "default": 1.5,
    "values": [0.5, 1.0, 1.5, 2.0]
  }
}
```

Behavior:

- `validation/validation_ast.py` and `validation/validation_cast.py` evaluate every entry in `values`.
- It saves the best result under `selected_params` in the validation report.
- `inference/llm_inference_steer_refusal.py` automatically loads those selected params at test time.
- If no validation report exists, inference falls back to `default`.
