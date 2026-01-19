# Troubleshooting & FAQ

## Frequently Asked Questions

### Q: What if the run is slow or stalls?
**A:** OpenAI API latency is the main factor. Lowering `TARGET_PER_CATEGORY` in your `.env` can help.

### Q: Why do I see "..." in the generated output?
**A:** LLM outputs are sometimes truncated due to token limits. Run `python3 repair_truncation.py` to fix these automatically.

### Q: How do I avoid overwriting my previous results?
**A:** The tool automatically archives the `outputs/` directory to `outputs_archive/` with a timestamp before each new run.

### Q: Can I use different models?
**A:** Yes. Set the `OPENAI_MODEL` environment variable or use the `--model` flag in scripts that support it.

## Common Issues

### YAML File Not Found
Ensure you are running the script from the repository root, or specify an absolute path to the YAML file.

### Missing Dependencies
If you encounter `ModuleNotFoundError`, ensure your virtual environment is active and you have run:
```bash
pip install openai python-dotenv jsonschema pandas openpyxl pyyaml
```
