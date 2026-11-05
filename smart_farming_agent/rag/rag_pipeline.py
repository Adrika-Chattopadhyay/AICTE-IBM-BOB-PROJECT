"""
RAG Pipeline — Retrieval-Augmented Generation for Smart Farming.
Uses FAISS vector store + IBM Granite for generation.

When IBM credentials are absent, operates in demo mode with
in-memory farming knowledge chunks.
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger
from smart_farming_agent.rag.granite_client import GraniteClient

settings = get_settings()

# ---------------------------------------------------------------------------
# Try importing vector store libraries — graceful degradation
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using keyword-based retrieval.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Using simple similarity search.")

# ---------------------------------------------------------------------------
# Built-in farming knowledge base (used when no external docs are loaded)
# ---------------------------------------------------------------------------
FARMING_KNOWLEDGE = [
    {
        "id": "kb_001",
        "topic": "kharif_crops",
        "content": (
            "Kharif crops are sown in June-July and harvested in September-October. "
            "Major kharif crops in India include rice, maize, jowar, bajra, cotton, "
            "soybean, groundnut, and sugarcane. These crops require warm and wet weather. "
            "Average rainfall of 750-1500mm is ideal for most kharif crops."
        ),
    },
    {
        "id": "kb_002",
        "topic": "rabi_crops",
        "content": (
            "Rabi crops are sown in October-November and harvested in March-April. "
            "Major rabi crops include wheat, barley, mustard, gram (chickpea), and peas. "
            "These crops grow in cool dry weather with moderate rainfall. "
            "Wheat requires 15-20°C during growing season."
        ),
    },
    {
        "id": "kb_003",
        "topic": "soil_types",
        "content": (
            "India has six major soil types: Alluvial soil (found in river plains, ideal for wheat, rice, sugarcane), "
            "Black soil / Regur (central India, ideal for cotton), Red and Yellow soil (eastern India, ideal for rice, wheat), "
            "Laterite soil (coastal areas, ideal for tea, coffee, rubber), Arid soil (Rajasthan, limited agriculture), "
            "Forest soil (hilly areas, tea, coffee). Each soil requires different fertilization strategies."
        ),
    },
    {
        "id": "kb_004",
        "topic": "irrigation",
        "content": (
            "Types of irrigation in India: Canal irrigation (35%), Well/Tube well irrigation (40%), "
            "Tank irrigation (15%), Drip and Sprinkler (10% and growing). "
            "Drip irrigation saves 30-50% water and is ideal for horticulture. "
            "Sprinkler irrigation is suitable for wheat, groundnut, and vegetables. "
            "PM KUSUM scheme provides 60-90% subsidy on solar-powered irrigation pumps."
        ),
    },
    {
        "id": "kb_005",
        "topic": "fertilizers",
        "content": (
            "Macronutrients: Nitrogen (N) promotes vegetative growth, Phosphorus (P) promotes root and flowering, "
            "Potassium (K) improves disease resistance and fruit quality. "
            "Recommended NPK for rice: 120:60:40 kg/ha. For wheat: 120:60:40 kg/ha. "
            "For cotton: 150:60:60 kg/ha. Urea contains 46% N, DAP contains 18% N and 46% P. "
            "Organic alternatives: FYM (Farm Yard Manure), compost, vermicompost, green manure."
        ),
    },
    {
        "id": "kb_006",
        "topic": "pest_management",
        "content": (
            "Integrated Pest Management (IPM) principles: Use resistant varieties, biological control, "
            "chemical control as last resort. Common pests: Stem borer (rice, sugarcane), "
            "Bollworm (cotton), Aphids (vegetables), White fly (cotton, vegetables). "
            "Biopesticides: Bt (Bacillus thuringiensis) effective against lepidoptera. "
            "Neem-based products: Azadirachtin effective against sucking pests."
        ),
    },
    {
        "id": "kb_007",
        "topic": "market_prices",
        "content": (
            "Minimum Support Price (MSP) for 2023-24: Wheat ₹2275/quintal, Rice ₹2183/quintal, "
            "Cotton (long staple) ₹7020/quintal, Groundnut ₹6377/quintal, Soybean ₹4600/quintal. "
            "Market prices vary by state and season. Agmarknet provides daily mandi prices. "
            "e-NAM (National Agriculture Market) connects farmers to buyers across India."
        ),
    },
    {
        "id": "kb_008",
        "topic": "government_schemes",
        "content": (
            "Key government schemes: PM-Kisan (₹6000/year income support), "
            "PM Fasal Bima Yojana (crop insurance at 1.5-5% premium), "
            "Kisan Credit Card (credit up to ₹3 lakh at 4% interest), "
            "Soil Health Card (free soil testing every 2 years), "
            "PM KUSUM (solar pump subsidy 60-90%), "
            "Paramparagat Krishi Vikas Yojana (organic farming promotion), "
            "National Horticulture Mission (subsidy for horticulture development)."
        ),
    },
    {
        "id": "kb_009",
        "topic": "weather_farming",
        "content": (
            "Weather impacts on farming: Drought stress at flowering reduces yield by 40-60%. "
            "Excess rainfall causes waterlogging and root rot. High humidity (>80%) promotes fungal diseases. "
            "Frost damages winter crops at temperatures below 0°C. "
            "El Niño years typically see below-normal monsoon in India. "
            "Kharif sowing window: June 15 to July 15 for most of India."
        ),
    },
    {
        "id": "kb_010",
        "topic": "organic_farming",
        "content": (
            "Organic farming principles: No synthetic chemicals, use of bio-inputs, crop rotation, "
            "cover cropping, and conservation tillage. "
            "Certification: NPOP (National Programme for Organic Production) mandatory for export. "
            "India has 3.76 million certified organic farmers (2023), largest in the world. "
            "Premium prices: 20-50% higher than conventional produce in domestic and export markets."
        ),
    },
    {
        "id": "kb_011",
        "topic": "crop_diseases",
        "content": (
            "Major crop diseases: Rice blast (Magnaporthe oryzae) - use Tricyclazole; "
            "Wheat rust (Puccinia spp.) - use Propiconazole; "
            "Late blight of potato/tomato (Phytophthora infestans) - use Mancozeb; "
            "Cotton bollrot - use Carbendazim; "
            "Citrus canker - use Copper oxychloride. "
            "Disease triangle: susceptible host + virulent pathogen + favourable environment = disease."
        ),
    },
    {
        "id": "kb_012",
        "topic": "seed_selection",
        "content": (
            "Certified seeds ensure 15-25% higher yield. Hybrid seeds have 20-30% yield advantage. "
            "Seed replacement rate should be 25-33% annually. "
            "Seed treatment: Thiram 3g/kg against seed-borne diseases, Imidacloprid for soil insects. "
            "Bio-seed treatment: Trichoderma viride 4g/kg + PSB 5g/kg + Azospirillum 5g/kg. "
            "Source certified seeds from government authorized dealers or Krishi Vigyan Kendras."
        ),
    },
]


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for farming queries.

    1. Retrieves relevant knowledge chunks via semantic or keyword search
    2. Builds a contextual prompt
    3. Generates a response using IBM Granite
    """

    SYSTEM_PROMPT = """You are KisanMitra, an expert AI agricultural advisor for Indian farmers.
You provide accurate, practical, localized farming advice based on scientific knowledge.
Your guidance covers: crop selection, soil management, weather-based advisories, 
pest/disease management, market prices, irrigation, fertilizers, and government schemes.

Rules:
- Always be specific to Indian agriculture context
- Give practical, actionable advice
- Mention government schemes when relevant
- Express quantities in Indian units (kg, quintal, hectare, acre)
- If uncertain, recommend consulting local KVK (Krishi Vigyan Kendra)
- Keep responses concise but complete (200-400 words ideal)
- Use simple language suitable for farmers
"""

    def __init__(self, granite_client: Optional[GraniteClient] = None):
        self._granite = granite_client or GraniteClient()
        self._knowledge_chunks = FARMING_KNOWLEDGE
        self._embedder = None
        self._index = None
        self._chunk_map: Dict[int, Dict] = {}
        self._embeddings_ready = False

        if EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE:
            self._init_vector_store()
        else:
            logger.info("Vector store unavailable — using keyword-based retrieval.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int = 3,
        language: str = "en",
        location: Optional[str] = None,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a farming question through the RAG pipeline.

        Returns:
            dict with keys: answer, sources, confidence, context_used
        """
        # 1. Retrieve relevant chunks
        chunks = self._retrieve(question, top_k=top_k)

        # 2. Build prompt with context
        context = self._build_context(chunks)
        prompt = self._build_prompt(
            question=question,
            context=context,
            location=location,
            season=season,
            language=language,
        )

        # 3. Generate response
        answer = self._granite.generate(prompt, max_tokens=min(1024, settings.max_tokens_per_request))

        # 4. Confidence based on retrieval quality
        confidence = self._compute_confidence(chunks, answer)

        return {
            "answer": answer,
            "sources": [c.get("topic", "knowledge_base") for c in chunks],
            "confidence": confidence,
            "context_used": len(chunks),
            "demo_mode": self._granite.is_demo_mode,
        }

    def add_document(self, content: str, topic: str, doc_id: Optional[str] = None) -> str:
        """Add a new document to the knowledge base."""
        chunk = {
            "id": doc_id or f"doc_{len(self._knowledge_chunks)+1:04d}",
            "topic": topic,
            "content": content,
        }
        self._knowledge_chunks.append(chunk)
        if self._embeddings_ready:
            self._add_to_index(chunk, len(self._knowledge_chunks) - 1)
        return chunk["id"]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self._embeddings_ready:
            return self._semantic_retrieve(query, top_k)
        return self._keyword_retrieve(query, top_k)

    def _semantic_retrieve(self, query: str, top_k: int) -> List[Dict]:
        try:
            import numpy as np
            q_vec = self._embedder.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_vec)
            scores, indices = self._index.search(q_vec, min(top_k, len(self._chunk_map)))
            return [
                self._chunk_map[idx]
                for idx in indices[0]
                if idx >= 0 and idx in self._chunk_map
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Semantic retrieval failed, falling back to keyword: {exc}")
            return self._keyword_retrieve(query, top_k)

    def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict]:
        query_words = set(query.lower().split())
        scored = []
        for chunk in self._knowledge_chunks:
            content_words = set(chunk["content"].lower().split())
            topic_words = set(chunk.get("topic", "").lower().replace("_", " ").split())
            overlap = len(query_words & (content_words | topic_words))
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:top_k]] or self._knowledge_chunks[:top_k]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No specific context available."
        parts = ["RELEVANT FARMING KNOWLEDGE:"]
        for i, chunk in enumerate(chunks, 1):
            topic = chunk.get("topic", "general").replace("_", " ").title()
            parts.append(f"\n[{i}] {topic}:\n{chunk['content']}")
        return "\n".join(parts)

    def _build_prompt(
        self,
        question: str,
        context: str,
        location: Optional[str],
        season: Optional[str],
        language: str,
    ) -> str:
        location_str = f"Farmer's location: {location}." if location else ""
        season_str = f"Current season: {season}." if season else ""
        lang_note = (
            f"The farmer asked in {language}. Respond in English with simple language."
            if language != "en"
            else ""
        )

        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"{context}\n\n"
            f"---\n"
            f"{location_str} {season_str} {lang_note}\n\n"
            f"Farmer's Question: {question}\n\n"
            f"Agricultural Expert Answer:"
        )

    # ------------------------------------------------------------------
    # Vector store initialization
    # ------------------------------------------------------------------
    def _init_vector_store(self) -> None:
        try:
            import numpy as np
            logger.info("Initializing sentence transformer embedder...")
            # Use a small, fast model
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            dim = self._embedder.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine with L2 norm)

            texts = [c["content"] for c in self._knowledge_chunks]
            embeddings = self._embedder.encode(texts, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)

            self._chunk_map = {i: chunk for i, chunk in enumerate(self._knowledge_chunks)}
            self._embeddings_ready = True
            logger.info(f"Vector store ready with {len(self._knowledge_chunks)} chunks.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Vector store init failed: {exc}. Using keyword retrieval.")
            self._embeddings_ready = False

    def _add_to_index(self, chunk: Dict, idx: int) -> None:
        try:
            import numpy as np
            vec = self._embedder.encode([chunk["content"]], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            self._index.add(vec)
            self._chunk_map[idx] = chunk
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to add chunk to vector store: {exc}")

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------
    def _compute_confidence(self, chunks: List[Dict], answer: str) -> float:
        if not chunks:
            return 0.40
        base = 0.70
        # Boost if answer contains words from the retrieved chunks
        all_chunk_words = set()
        for c in chunks:
            all_chunk_words.update(c["content"].lower().split())
        answer_words = set(answer.lower().split())
        overlap_ratio = len(answer_words & all_chunk_words) / max(len(answer_words), 1)
        return min(1.0, base + overlap_ratio * 0.30)
