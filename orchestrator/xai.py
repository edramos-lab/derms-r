"""AI-generated explanations (XAI) for each Exploratory Dataset Analysis
section, via OpenRouter's chat completions API (DeepSeek by default).

The API key lives in OPENROUTER_API_KEY (see repo root .env, gitignored —
never hardcoded, never sent to the browser; only this server-side module
reads it). If it's missing or the call fails, callers get a clear error
string instead of a stack trace: the EDA page works fully without this
feature, it's an enrichment, not a dependency.
"""
import json
import os
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-chat"

SECTION_TOPICS = {
    "metrics": "the dataset-level class-balance and image-size statistics",
    "distribution": "the class distribution bar chart and its entropy-balance reference line",
    "rgb": "the RGB channel histogram and the color/brightness/contrast summary",
    "crossval": "the K-Fold / Stratified K-Fold cross-validation split visualization",
    "pca": "the 2D PCA projection of a per-image mean-RGB feature",
    "tsne": "the 3D t-SNE projection of a per-image mean-RGB feature",
}


def explain_section(section: str, dataset_id: str, context: dict) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY is not configured on the orchestrator container."}
    topic = SECTION_TOPICS.get(section, section)
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    prompt = (
        f"You are assisting a machine learning engineer reading an Exploratory Data Analysis "
        f"report for the image dataset '{dataset_id}'. Explain, in 3-5 concise sentences, what "
        f"{topic} implies for training an image classifier on this dataset. Be specific to the "
        f"numbers given below, not generic advice, and do not repeat the raw numbers back "
        f"verbatim — interpret them.\n\nData:\n{json.dumps(context, indent=2, default=str)}"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 320,
        "temperature": 0.3,
    }).encode()
    request = urllib.request.Request(
        OPENROUTER_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "RIGAL-AI",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        return {"error": f"OpenRouter returned {error.code}: {detail}"}
    except urllib.error.URLError as error:
        return {"error": f"Could not reach OpenRouter: {error.reason}"}
    except TimeoutError:
        return {"error": "OpenRouter request timed out after 30s."}
    try:
        text = payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return {"error": "Unexpected response shape from OpenRouter."}
    return {"text": text, "model": model}


def section_context(section: str, insights: dict) -> dict:
    """Extract the numeric/textual subset relevant to one section — never
    image bytes, keeps the prompt small and avoids sending plot pixels to a
    third-party API."""
    if section == "metrics":
        return insights.get("metrics", {})
    if section == "distribution":
        return {
            "class_counts": insights.get("class_counts"),
            "entropy_balance": insights.get("metrics", {}).get("entropy_balance"),
            "imbalance_ratio": insights.get("metrics", {}).get("imbalance_ratio"),
        }
    if section == "rgb":
        return insights.get("rgb_insights", {}) or {}
    if section == "crossval":
        return {
            "k": insights.get("crossval", {}).get("k"),
            "strategy": insights.get("crossval", {}).get("strategy"),
            "class_counts": insights.get("class_counts"),
        }
    if section == "pca":
        return {
            "explained_variance": insights.get("pca_explained_variance"),
            "sample_size": insights.get("sample_size"),
            "num_classes": insights.get("metrics", {}).get("num_classes"),
        }
    if section == "tsne":
        return {
            "sample_size": insights.get("sample_size"),
            "num_classes": insights.get("metrics", {}).get("num_classes"),
        }
    return {}
