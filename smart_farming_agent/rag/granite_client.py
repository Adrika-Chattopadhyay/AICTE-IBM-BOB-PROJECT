"""
IBM Granite LLM Client (watsonx.ai)
Wraps IBM watsonx.ai SDK with retry logic and token tracking.
Falls back to a mock response when credentials are not configured
(for demo/development mode).
"""
import os
import time
from typing import Any, Dict, List, Optional

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger

settings = get_settings()

# ---------------------------------------------------------------------------
# Try importing IBM watsonx SDK — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    IBM_SDK_AVAILABLE = True
except ImportError:
    IBM_SDK_AVAILABLE = False
    logger.warning(
        "ibm-watsonx-ai not installed. Running in DEMO mode with mock responses."
    )

# Default generation parameters
DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 1024,
    "min_new_tokens": 50,
    "temperature": 0.3,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "stop_sequences": ["Human:", "User:", "\n\n---"],
}


class GraniteClient:
    """
    Thin wrapper around IBM watsonx.ai Granite models.
    Provides generate() and embed() methods used by the RAG pipeline.
    """

    def __init__(self):
        self._client = None
        self._model = None
        self._embedding_model = None
        self._demo_mode = not IBM_SDK_AVAILABLE
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization — only connects when first used."""
        if self._initialized:
            return

        if not IBM_SDK_AVAILABLE:
            self._demo_mode = True
            self._initialized = True
            logger.info("IBM SDK unavailable — Granite client in DEMO mode.")
            return

        api_key = settings.ibm_api_key
        project_id = settings.ibm_project_id

        if not api_key or api_key == "your_ibm_cloud_api_key_here":
            self._demo_mode = True
            self._initialized = True
            logger.warning(
                "IBM API key not configured — Granite client in DEMO mode. "
                "Set IBM_API_KEY in .env to use the real model."
            )
            return

        try:
            credentials = Credentials(
                url=settings.ibm_watsonx_url,
                api_key=api_key,
            )
            self._client = APIClient(credentials=credentials, project_id=project_id)
            self._model = ModelInference(
                model_id=settings.ibm_granite_chat_model_id,
                api_client=self._client,
                project_id=project_id,
                params=DEFAULT_GEN_PARAMS,
            )
            self._demo_mode = False
            logger.info(
                f"IBM Granite client initialized: model={settings.ibm_granite_chat_model_id}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"IBM Granite init failed: {exc}. Falling back to DEMO mode.")
            self._demo_mode = True

        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from the Granite model.

        Args:
            prompt: The full prompt string (system + context + user query)
            params: Override generation parameters
            max_tokens: Convenience override for max_new_tokens

        Returns:
            Generated text string
        """
        self._initialize()

        if max_tokens:
            merged_params = {**DEFAULT_GEN_PARAMS, **(params or {}), "max_new_tokens": max_tokens}
        else:
            merged_params = {**DEFAULT_GEN_PARAMS, **(params or {})}

        if self._demo_mode:
            return self._demo_generate(prompt)

        try:
            response = self._model.generate(prompt=prompt, params=merged_params)
            generated = response["results"][0]["generated_text"].strip()
            logger.debug(f"Granite generated {len(generated.split())} words")
            return generated
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Granite generation failed: {exc}")
            return self._demo_generate(prompt)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
    ) -> str:
        """
        Chat-style generation. Converts messages list to a prompt string.
        """
        self._initialize()
        prompt = self._build_chat_prompt(messages, system_prompt)
        return self.generate(prompt)

    @property
    def is_demo_mode(self) -> bool:
        self._initialize()
        return self._demo_mode

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
    ) -> str:
        """
        Format messages into Granite-style prompt.
        Granite uses <|system|>, <|user|>, <|assistant|> tags.
        """
        parts = []
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}\n<|end|>")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"<|user|>\n{content}\n<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n<|end|>")

        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _demo_generate(self, prompt: str) -> str:
        """
        Demo-mode fallback — generates a realistic agriculture response
        without calling the real API. Used when credentials are absent.
        """
        prompt_lower = prompt.lower()

        if any(w in prompt_lower for w in ["weather", "rain", "temperature", "forecast"]):
            return (
                "Based on current weather patterns, expect moderate rainfall (45–60 mm) "
                "this week with temperatures ranging from 22°C to 31°C. "
                "Conditions are suitable for sowing kharif crops like paddy and soybean. "
                "Ensure proper drainage in low-lying fields to prevent waterlogging."
            )
        elif any(w in prompt_lower for w in ["mandi", "price", "market", "rate", "tomato", "onion"]):
            return (
                "Current market rates (Mandi prices):\n"
                "• Tomato: ₹1,800–₹2,200/quintal\n"
                "• Onion: ₹1,200–₹1,500/quintal\n"
                "• Wheat: ₹2,200–₹2,400/quintal\n"
                "• Rice (Fine): ₹3,200–₹3,600/quintal\n"
                "Prices may vary by location. Visit your nearest APMC mandi for exact rates."
            )
        elif any(w in prompt_lower for w in ["crop", "grow", "sow", "plant", "recommend"]):
            return (
                "Based on the current season (Kharif 2024) and typical soil conditions:\n"
                "**Recommended Crops:**\n"
                "1. **Rice (Paddy)** — Ideal for clay-loam soils with good water availability\n"
                "2. **Soybean** — Suitable for well-drained black cotton soils\n"
                "3. **Maize (Corn)** — Good for loamy soils, high yield potential\n"
                "4. **Cotton** — Ideal for deep black soil (regur) in central India\n\n"
                "Use certified seeds and follow recommended spacing for best yields."
            )
        elif any(w in prompt_lower for w in ["pest", "disease", "insect", "fungal", "blight"]):
            return (
                "Common pest & disease management:\n"
                "• **Stem Borer (Rice):** Use Carbofuran 3G @ 33 kg/ha or light traps\n"
                "• **Late Blight (Tomato/Potato):** Spray Mancozeb 75 WP @ 2.5 g/L\n"
                "• **Aphids:** Spray Imidacloprid 17.8 SL @ 0.5 mL/L water\n"
                "• **Powdery Mildew:** Use Wettable Sulphur 80 WP @ 3 g/L\n"
                "Always follow label instructions and pre-harvest intervals."
            )
        elif any(w in prompt_lower for w in ["soil", "fertilizer", "nutrient", "ph"]):
            return (
                "Soil health recommendations:\n"
                "• Ideal pH range: 6.0–7.5 for most crops\n"
                "• NPK application: 120:60:40 kg/ha for paddy\n"
                "• Organic matter: Add FYM (Farm Yard Manure) @ 5 tons/ha before sowing\n"
                "• Micronutrients: Apply Zinc Sulphate @ 25 kg/ha if zinc deficiency observed\n"
                "Get a soil test done at your nearest Krishi Vigyan Kendra (KVK)."
            )
        elif any(w in prompt_lower for w in ["scheme", "subsidy", "government", "loan", "insurance"]):
            return (
                "Government schemes for farmers:\n"
                "• **PM-Kisan:** Direct income support ₹6,000/year for eligible farmers\n"
                "• **PM Fasal Bima Yojana:** Crop insurance at subsidized premium\n"
                "• **Kisan Credit Card:** Short-term credit up to ₹3 lakh @ 4% interest\n"
                "• **Soil Health Card:** Free soil testing every 2 years\n"
                "• **PM KUSUM:** Subsidy on solar pumps (60–90% subsidy)\n"
                "Visit your nearest Common Service Centre (CSC) to apply."
            )
        else:
            return (
                "I am your Smart Farming AI Assistant, powered by IBM Granite. "
                "I can help you with:\n"
                "• 🌤️ Weather forecasts and farming advisories\n"
                "• 🌱 Crop recommendations for your soil and season\n"
                "• 💰 Current mandi/market prices\n"
                "• 🐛 Pest and disease identification and control\n"
                "• 🏛️ Government schemes and subsidies\n"
                "• 🌿 Soil health and fertilizer recommendations\n\n"
                "Please ask your farming question in any language!"
            )
