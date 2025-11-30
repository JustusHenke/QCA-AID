# Provider Fallback Configurations

This directory contains local fallback configurations for LLM providers. These files are used when the Catwalk GitHub repository is unreachable.

## Files

- **openai.json** - OpenAI provider configuration with model metadata
- **anthropic.json** - Anthropic provider configuration with model metadata  
- **openrouter.json** - OpenRouter provider configuration with model metadata

## Purpose

The LLM Provider Manager attempts to load provider configurations from the Catwalk GitHub repository:
```
https://raw.githubusercontent.com/charmbracelet/catwalk/main/internal/providers/configs/
```

If the GitHub URL is unreachable (due to network issues, rate limiting, or offline development), the system automatically falls back to these local configuration files.

## Updating Configurations

To update the fallback configurations with the latest data from Catwalk, run:

```bash
python download_fallback_configs.py
```

This script will:
1. Download the latest configurations from Catwalk
2. Save them to this directory
3. Overwrite existing files

## Configuration Format

Each configuration file follows the Catwalk provider schema:

```json
{
  "name": "Provider Name",
  "id": "provider-id",
  "type": "provider-type",
  "api_key": "$ENV_VAR_NAME",
  "api_endpoint": "https://api.example.com",
  "default_large_model_id": "model-id",
  "default_small_model_id": "model-id",
  "models": [
    {
      "id": "model-id",
      "name": "Model Name",
      "cost_per_1m_in": 1.0,
      "cost_per_1m_out": 2.0,
      "context_window": 128000,
      "can_reason": false,
      "supports_attachments": false
    }
  ]
}
```

## Requirements

These fallback configurations satisfy **Requirement 2.2**:
> WHEN the GitHub-URL is not reachable THEN the system SHALL fall back to local copies of provider configurations

## Last Updated

These configurations were last downloaded from Catwalk on: **2024-11-30**

To check the current version in Catwalk, visit:
- [OpenAI Config](https://github.com/charmbracelet/catwalk/blob/main/internal/providers/configs/openai.json)
- [Anthropic Config](https://github.com/charmbracelet/catwalk/blob/main/internal/providers/configs/anthropic.json)
- [OpenRouter Config](https://github.com/charmbracelet/catwalk/blob/main/internal/providers/configs/openrouter.json)
