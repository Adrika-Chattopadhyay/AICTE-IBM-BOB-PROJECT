"""
Farming Tools — External API integrations with caching and fallback.
Each tool follows a consistent interface: call() -> dict

Tools:
  - WeatherTool: OpenWeatherMap API
  - MarketPriceTool: Agmarknet / data.gov.in API
  - SoilTool: ISRIC SoilGrids API
  - CropRecommendationTool: Rule-based crop engine
  - PestControlTool: Pest diagnosis and treatment lookup
  - GovernmentSchemeTool: Scheme information lookup
"""
import json
import time
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger

settings = get_settings()

# Simple in-memory cache: key -> (data, expire_at)
_CACHE: Dict[str, tuple] = {}
CACHE_TTL = 3600  # 1 hour default


def _cache_get(key: str) -> Optional[Any]:
    if key in _CACHE:
        data, expire_at = _CACHE[key]
        if time.monotonic() < expire_at:
            return data
        del _CACHE[key]
    return None


def _cache_set(key: str, data: Any, ttl: int = CACHE_TTL) -> None:
    _CACHE[key] = (data, time.monotonic() + ttl)


def _make_cache_key(*args) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()


# ===========================================================================
# Base Tool
# ===========================================================================
class BaseTool(ABC):
    name: str = "base_tool"
    description: str = ""
    timeout: int = 15

    @abstractmethod
    def call(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool and return a standardized result dict."""
        ...

    def _get(self, url: str, params: Dict = None, headers: Dict = None) -> requests.Response:
        """Shared HTTP GET with timeout and error handling."""
        try:
            resp = requests.get(
                url, params=params, headers=headers or {},
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Tool '{self.name}' request timed out after {self.timeout}s.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Tool '{self.name}' HTTP error: {e}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Tool '{self.name}' could not connect to API.")


# ===========================================================================
# Weather Tool
# ===========================================================================
class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather and 5-day forecast for a location in India."

    WEATHER_CODES = {
        200: "Thunderstorm", 300: "Drizzle", 500: "Rain", 600: "Snow",
        700: "Mist/Fog", 800: "Clear Sky", 801: "Few Clouds",
        802: "Scattered Clouds", 803: "Broken Clouds", 804: "Overcast Clouds",
    }

    def call(self, location: str, units: str = "metric") -> Dict[str, Any]:
        """
        Args:
            location: City name, e.g. 'Pune' or 'Nashik,IN'
            units: 'metric' (Celsius) or 'imperial'
        """
        cache_key = _make_cache_key("weather", location, units)
        cached = _cache_get(cache_key)
        if cached:
            logger.debug(f"Weather cache hit for '{location}'")
            return cached

        api_key = settings.openweather_api_key
        if not api_key or api_key == "your_openweathermap_api_key_here":
            return self._demo_weather(location)

        try:
            url = f"{settings.openweather_base_url}/weather"
            resp = self._get(url, params={
                "q": f"{location},{settings.default_country_code}",
                "appid": api_key,
                "units": units,
            })
            raw = resp.json()
            result = {
                "location": raw.get("name", location),
                "temperature_c": raw["main"]["temp"],
                "feels_like_c": raw["main"]["feels_like"],
                "humidity_pct": raw["main"]["humidity"],
                "description": raw["weather"][0]["description"].title(),
                "wind_speed_kmh": round(raw["wind"]["speed"] * 3.6, 1),
                "visibility_km": raw.get("visibility", 0) / 1000,
                "pressure_hpa": raw["main"]["pressure"],
                "sunrise": datetime.fromtimestamp(raw["sys"]["sunrise"]).strftime("%H:%M"),
                "sunset": datetime.fromtimestamp(raw["sys"]["sunset"]).strftime("%H:%M"),
                "farming_advisory": self._farming_advisory(raw),
                "source": "OpenWeatherMap",
                "timestamp": datetime.now().isoformat(),
            }
            _cache_set(cache_key, result, ttl=1800)  # 30 min for weather
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Weather API error: {exc}. Using demo data.")
            return self._demo_weather(location)

    def _farming_advisory(self, raw: dict) -> str:
        temp = raw["main"]["temp"]
        humidity = raw["main"]["humidity"]
        weather_id = raw["weather"][0]["id"]
        advisories = []
        if weather_id >= 200 and weather_id < 300:
            advisories.append("⚠️ Thunderstorms expected. Avoid field operations.")
        elif weather_id >= 500 and weather_id < 600:
            advisories.append("🌧️ Rain expected. Ensure proper field drainage.")
        if temp > 38:
            advisories.append("🔥 High temperature. Increase irrigation frequency.")
        if temp < 10:
            advisories.append("❄️ Cold conditions. Protect sensitive crops from frost.")
        if humidity > 85:
            advisories.append("💧 High humidity. Watch for fungal diseases.")
        if not advisories:
            advisories.append("✅ Good farming conditions today.")
        return " | ".join(advisories)

    def _demo_weather(self, location: str) -> Dict[str, Any]:
        return {
            "location": location,
            "temperature_c": 28.5,
            "feels_like_c": 31.2,
            "humidity_pct": 72,
            "description": "Partly Cloudy",
            "wind_speed_kmh": 12.5,
            "visibility_km": 8.0,
            "pressure_hpa": 1012,
            "sunrise": "06:15",
            "sunset": "18:45",
            "farming_advisory": "✅ Good farming conditions today. Moderate humidity — monitor for fungal issues.",
            "source": "Demo Data",
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True,
        }


# ===========================================================================
# Market Price Tool
# ===========================================================================
class MarketPriceTool(BaseTool):
    name = "market_price"
    description = "Get current mandi/market prices for agricultural commodities."

    # Demo price database (realistic 2024 prices in INR/quintal)
    DEMO_PRICES = {
        "tomato": {"min": 1200, "max": 2500, "modal": 1800, "unit": "quintal"},
        "onion": {"min": 800, "max": 1800, "modal": 1200, "unit": "quintal"},
        "potato": {"min": 600, "max": 1400, "modal": 950, "unit": "quintal"},
        "wheat": {"min": 2200, "max": 2600, "modal": 2375, "unit": "quintal"},
        "rice": {"min": 2800, "max": 3800, "modal": 3200, "unit": "quintal"},
        "maize": {"min": 1700, "max": 2200, "modal": 1900, "unit": "quintal"},
        "cotton": {"min": 6200, "max": 7500, "modal": 6800, "unit": "quintal"},
        "soybean": {"min": 4200, "max": 5000, "modal": 4600, "unit": "quintal"},
        "groundnut": {"min": 5500, "max": 6800, "modal": 6200, "unit": "quintal"},
        "sugarcane": {"min": 290, "max": 340, "modal": 315, "unit": "quintal"},
        "banana": {"min": 800, "max": 1600, "modal": 1100, "unit": "quintal"},
        "mango": {"min": 2000, "max": 5000, "modal": 3200, "unit": "quintal"},
        "chilli": {"min": 4000, "max": 8000, "modal": 5500, "unit": "quintal"},
        "garlic": {"min": 2000, "max": 4000, "modal": 2800, "unit": "quintal"},
        "mustard": {"min": 5000, "max": 6000, "modal": 5500, "unit": "quintal"},
    }

    def call(self, commodity: str, state: str = "Maharashtra", market: str = "") -> Dict[str, Any]:
        cache_key = _make_cache_key("market", commodity, state, market)
        cached = _cache_get(cache_key)
        if cached:
            return cached

        api_key = settings.data_gov_api_key
        if not api_key or api_key == "your_data_gov_in_api_key_here":
            return self._demo_price(commodity, state, market)

        try:
            resp = self._get(
                settings.agmarknet_api_url,
                params={
                    "api-key": api_key,
                    "format": "json",
                    "filters[commodity]": commodity.capitalize(),
                    "filters[state]": state,
                    "limit": 10,
                },
            )
            records = resp.json().get("records", [])
            if not records:
                return self._demo_price(commodity, state, market)

            result = {
                "commodity": commodity,
                "state": state,
                "records": records[:5],
                "source": "Agmarknet / data.gov.in",
                "timestamp": datetime.now().isoformat(),
            }
            _cache_set(cache_key, result, ttl=3600)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Market price API error: {exc}. Using demo data.")
            return self._demo_price(commodity, state, market)

    def _demo_price(self, commodity: str, state: str, market: str) -> Dict[str, Any]:
        key = commodity.lower().strip()
        prices = self.DEMO_PRICES.get(key, {
            "min": 1000, "max": 2000, "modal": 1500, "unit": "quintal"
        })
        return {
            "commodity": commodity.title(),
            "state": state,
            "market": market or "Local APMC",
            "min_price_inr": prices["min"],
            "max_price_inr": prices["max"],
            "modal_price_inr": prices["modal"],
            "unit": prices["unit"],
            "msp_if_applicable": self._get_msp(key),
            "source": "Demo Data (Indicative)",
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True,
        }

    def _get_msp(self, commodity: str) -> Optional[int]:
        MSP_2024 = {
            "wheat": 2275, "rice": 2183, "maize": 2090,
            "soybean": 4600, "groundnut": 6377, "cotton": 7020,
            "mustard": 5650, "sugarcane": 315,
        }
        return MSP_2024.get(commodity)


# ===========================================================================
# Soil Tool
# ===========================================================================
class SoilTool(BaseTool):
    name = "soil"
    description = "Get soil composition and health data for a geographic location."

    def call(self, latitude: float, longitude: float, depth: str = "0-30cm") -> Dict[str, Any]:
        cache_key = _make_cache_key("soil", latitude, longitude, depth)
        cached = _cache_get(cache_key)
        if cached:
            return cached

        try:
            url = f"{settings.soilgrids_api_url}/properties/query"
            resp = self._get(url, params={
                "lon": longitude,
                "lat": latitude,
                "property": ["phh2o", "nitrogen", "clay", "sand", "silt", "soc"],
                "depth": depth,
                "value": "mean",
            })
            data = resp.json()
            result = self._parse_soilgrids(data, latitude, longitude)
            _cache_set(cache_key, result, ttl=86400)  # 24 hours for soil
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SoilGrids API error: {exc}. Using demo data.")
            return self._demo_soil(latitude, longitude)

    def _parse_soilgrids(self, data: dict, lat: float, lon: float) -> Dict[str, Any]:
        try:
            props = data.get("properties", {}).get("layers", {})
            return {
                "latitude": lat, "longitude": lon,
                "ph": round(props.get("phh2o", {}).get("depths", [{}])[0].get("values", {}).get("mean", 65) / 10, 1),
                "nitrogen_cg_kg": props.get("nitrogen", {}).get("depths", [{}])[0].get("values", {}).get("mean", 150),
                "clay_pct": props.get("clay", {}).get("depths", [{}])[0].get("values", {}).get("mean", 25),
                "sand_pct": props.get("sand", {}).get("depths", [{}])[0].get("values", {}).get("mean", 40),
                "silt_pct": props.get("silt", {}).get("depths", [{}])[0].get("values", {}).get("mean", 35),
                "source": "ISRIC SoilGrids",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception:
            return self._demo_soil(lat, lon)

    def _demo_soil(self, lat: float, lon: float) -> Dict[str, Any]:
        # Approximate soil type based on Indian geography
        soil_type = "Alluvial"
        if 18 <= lat <= 24 and 73 <= lon <= 80:
            soil_type = "Black Cotton (Vertisol)"
        elif lat < 15:
            soil_type = "Red and Laterite"
        elif lon > 80:
            soil_type = "Alluvial (Coastal)"

        return {
            "latitude": lat, "longitude": lon,
            "soil_type": soil_type,
            "ph": 6.8,
            "nitrogen_kg_ha": 180,
            "phosphorus_kg_ha": 22,
            "potassium_kg_ha": 280,
            "organic_carbon_pct": 0.65,
            "clay_pct": 28,
            "sand_pct": 38,
            "silt_pct": 34,
            "water_holding_capacity": "Moderate",
            "drainage": "Well-drained",
            "recommendations": [
                "Soil pH is in optimal range (6.5–7.5)",
                "Add organic matter to improve water retention",
                "Nitrogen supplementation recommended before sowing",
            ],
            "source": "Demo Data (Estimated)",
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True,
        }


# ===========================================================================
# Crop Recommendation Tool
# ===========================================================================
class CropRecommendationTool(BaseTool):
    name = "crop"
    description = "Get crop recommendations based on soil type, season, and location."

    CROP_MATRIX = {
        ("kharif", "Alluvial"): ["Rice", "Maize", "Soybean", "Sugarcane"],
        ("kharif", "Black Cotton (Vertisol)"): ["Cotton", "Soybean", "Jowar", "Maize"],
        ("kharif", "Red and Laterite"): ["Groundnut", "Maize", "Ragi", "Arhar"],
        ("kharif", "Sandy Loam"): ["Groundnut", "Moong", "Bajra", "Sesame"],
        ("rabi", "Alluvial"): ["Wheat", "Mustard", "Gram", "Peas"],
        ("rabi", "Black Cotton (Vertisol)"): ["Wheat", "Gram", "Linseed", "Safflower"],
        ("rabi", "Red and Laterite"): ["Wheat", "Sunflower", "Gram", "Potato"],
        ("rabi", "Sandy Loam"): ["Mustard", "Gram", "Barley", "Wheat"],
        ("zaid", "Alluvial"): ["Watermelon", "Muskmelon", "Cucumber", "Moong"],
        ("zaid", "Sandy Loam"): ["Watermelon", "Bitter Gourd", "Moong"],
    }

    def call(self, soil_type: str, season: str, state: str = "Maharashtra",
             rainfall_mm: float = 700, ph: float = 6.8) -> Dict[str, Any]:
        season_lower = season.lower().strip()
        soil_key = None
        for k in self.CROP_MATRIX:
            if k[0] == season_lower and k[1].lower() in soil_type.lower():
                soil_key = k
                break
        if not soil_key:
            # Default to alluvial
            soil_key = (season_lower if season_lower in ("kharif", "rabi", "zaid") else "kharif", "Alluvial")

        crops = self.CROP_MATRIX.get(soil_key, ["Rice", "Wheat", "Maize"])

        # Adjust for rainfall
        if rainfall_mm < 500:
            crops = [c for c in crops if c not in ["Rice", "Sugarcane"]]
            if not crops:
                crops = ["Bajra", "Jowar", "Groundnut"]
        if ph < 5.5:
            crops.insert(0, "Rice")  # Rice tolerates acidic soil
        elif ph > 7.5:
            crops = [c for c in crops if c not in ["Rice"]]

        return {
            "recommended_crops": crops[:5],
            "season": season,
            "soil_type": soil_type,
            "state": state,
            "sowing_window": self._sowing_window(season_lower),
            "notes": self._crop_notes(crops[:3]),
            "source": "Agricultural Research Data",
            "timestamp": datetime.now().isoformat(),
        }

    def _sowing_window(self, season: str) -> str:
        windows = {
            "kharif": "June 15 – July 15",
            "rabi": "October 15 – November 30",
            "zaid": "February 15 – March 31",
        }
        return windows.get(season, "Consult local KVK")

    def _crop_notes(self, crops: List[str]) -> str:
        notes = {
            "Rice": "Use SRI method to save 30% water.",
            "Wheat": "Use HD-2967 or GW-322 varieties for high yield.",
            "Cotton": "Bt cotton recommended for bollworm resistance.",
            "Soybean": "JS 335 or MACS 450 are popular varieties.",
            "Maize": "Use hybrid seeds for 30% higher yield.",
        }
        return " | ".join(notes[c] for c in crops if c in notes) or "Use certified seeds."


# ===========================================================================
# Pest Control Tool
# ===========================================================================
class PestControlTool(BaseTool):
    name = "pest_control"
    description = "Identify pests and diseases and recommend control measures."

    PEST_DB = {
        ("rice", "yellowing leaves"): {
            "pest": "Yellow Stem Borer (Scirpophaga incertulas)",
            "severity": "High",
            "chemical": "Cartap Hydrochloride 4G @ 18 kg/ha or Chlorpyrifos 20 EC @ 2 mL/L",
            "biological": "Release Trichogramma japonicum (50,000/ha) at egg-laying stage",
            "cultural": "Clip tips of seedlings before transplanting; avoid late planting",
        },
        ("rice", "brown spots"): {
            "pest": "Brown Spot (Bipolaris oryzae)",
            "severity": "Moderate",
            "chemical": "Mancozeb 75 WP @ 2.5 g/L or Edifenphos 50 EC @ 1 mL/L",
            "biological": "Seed treatment with Trichoderma viride 4 g/kg",
            "cultural": "Balanced fertilization; avoid water stress",
        },
        ("tomato", "wilting"): {
            "pest": "Fusarium Wilt (Fusarium oxysporum)",
            "severity": "High",
            "chemical": "Carbendazim 50 WP @ 2 g/L (drench around roots)",
            "biological": "Apply Trichoderma harzianum 2.5 kg/ha with FYM",
            "cultural": "Use resistant varieties; crop rotation with cereals",
        },
        ("cotton", "leaf curl"): {
            "pest": "Cotton Leaf Curl Virus (CLCuV) via Whitefly",
            "severity": "Very High",
            "chemical": "Imidacloprid 17.8 SL @ 0.3 mL/L or Thiamethoxam 25 WG @ 0.3 g/L",
            "biological": "Yellow sticky traps (10/ha); neem oil 1500 ppm @ 5 mL/L",
            "cultural": "Use tolerant varieties; remove and destroy infected plants",
        },
        ("wheat", "orange powder"): {
            "pest": "Wheat Rust (Puccinia spp.)",
            "severity": "High",
            "chemical": "Propiconazole 25 EC @ 0.1% or Tebuconazole 25.9 EC @ 0.1%",
            "biological": "Bioagent Bacillus subtilis (2 mL/L)",
            "cultural": "Grow resistant varieties; early sowing",
        },
    }

    def call(self, crop: str, symptoms: str) -> Dict[str, Any]:
        crop_lower = crop.lower().strip()
        symptoms_lower = symptoms.lower().strip()

        best_match = None
        best_score = 0
        for (c, s), info in self.PEST_DB.items():
            if c in crop_lower:
                score = sum(1 for word in s.split() if word in symptoms_lower)
                if score > best_score:
                    best_score = score
                    best_match = info

        if not best_match:
            best_match = {
                "pest": "Unknown pest/disease",
                "severity": "Unknown",
                "chemical": "Consult your nearest Krishi Vigyan Kendra (KVK) for diagnosis.",
                "biological": "Consider using broad-spectrum biopesticides as first line of defense.",
                "cultural": "Remove infected plant material and maintain field hygiene.",
            }

        return {
            "crop": crop,
            "reported_symptoms": symptoms,
            "diagnosis": best_match,
            "general_advice": (
                "Always follow label instructions. Observe pre-harvest intervals. "
                "Spray in morning or evening to avoid drift and heat-related losses."
            ),
            "source": "Agricultural Pest Management Database",
            "timestamp": datetime.now().isoformat(),
        }


# ===========================================================================
# Government Scheme Tool
# ===========================================================================
class GovernmentSchemeTool(BaseTool):
    name = "scheme_info"
    description = "Get information about government agricultural schemes and subsidies."

    SCHEMES = [
        {
            "name": "PM-Kisan Samman Nidhi",
            "benefit": "₹6,000/year (₹2,000 × 3 installments)",
            "eligibility": "Small and marginal farmers with less than 2 hectares land",
            "how_to_apply": "Apply at Common Service Centre or pmkisan.gov.in",
            "documents": ["Aadhaar card", "Land records", "Bank account"],
            "category": "income_support",
        },
        {
            "name": "PM Fasal Bima Yojana (PMFBY)",
            "benefit": "Crop insurance with 1.5% premium for Rabi, 2% for Kharif",
            "eligibility": "All farmers growing notified crops",
            "how_to_apply": "Apply through nearest bank branch or CSC before cut-off date",
            "documents": ["Land records", "Aadhaar", "Bank account", "Sowing certificate"],
            "category": "insurance",
        },
        {
            "name": "Kisan Credit Card (KCC)",
            "benefit": "Revolving credit up to ₹3 lakh at 4% interest per annum",
            "eligibility": "All farmers, sharecroppers, and SHG members",
            "how_to_apply": "Apply at nearest bank branch",
            "documents": ["Aadhaar", "Land records", "Passport photo"],
            "category": "credit",
        },
        {
            "name": "Soil Health Card Scheme",
            "benefit": "Free soil testing every 2 years with fertilizer recommendations",
            "eligibility": "All farmers",
            "how_to_apply": "Contact nearest Agriculture Department office or KVK",
            "documents": ["Land details", "Aadhaar"],
            "category": "soil_health",
        },
        {
            "name": "PM KUSUM (Solar Pump Scheme)",
            "benefit": "60-90% subsidy on solar-powered irrigation pumps",
            "eligibility": "Farmers with valid irrigation source",
            "how_to_apply": "Apply through State Nodal Agency (DISCOM/Agriculture Dept)",
            "documents": ["Aadhaar", "Land records", "Water source proof"],
            "category": "irrigation",
        },
        {
            "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
            "benefit": "₹50,000/ha over 3 years for organic farming transition",
            "eligibility": "Farmers willing to adopt organic farming in clusters",
            "how_to_apply": "Apply through Agriculture Department / FPO",
            "documents": ["Land records", "Aadhaar"],
            "category": "organic",
        },
        {
            "name": "Rashtriya Krishi Vikas Yojana (RKVY)",
            "benefit": "Financial assistance for infrastructure and value chain projects",
            "eligibility": "Farmers, FPOs, cooperatives, agri-startups",
            "how_to_apply": "Apply through State Agriculture Department",
            "documents": ["Project proposal", "Land records"],
            "category": "infrastructure",
        },
    ]

    def call(self, state: str = "Maharashtra", category: str = "", keyword: str = "") -> Dict[str, Any]:
        results = self.SCHEMES

        if category:
            results = [s for s in results if category.lower() in s.get("category", "").lower()]

        if keyword:
            kw_lower = keyword.lower()
            results = [
                s for s in results
                if kw_lower in s["name"].lower()
                or kw_lower in s["benefit"].lower()
                or kw_lower in s.get("category", "").lower()
            ]

        return {
            "state": state,
            "schemes": results[:6],
            "total_found": len(results),
            "note": (
                "Scheme eligibility and benefits may vary by state. "
                "Visit your nearest Krishi Vigyan Kendra or Common Service Centre for details."
            ),
            "source": "Government of India Agricultural Schemes Database",
            "timestamp": datetime.now().isoformat(),
        }


# ===========================================================================
# Tool registry
# ===========================================================================
TOOL_INSTANCES: Dict[str, BaseTool] = {
    "weather": WeatherTool(),
    "market_price": MarketPriceTool(),
    "soil": SoilTool(),
    "crop": CropRecommendationTool(),
    "pest_control": PestControlTool(),
    "scheme_info": GovernmentSchemeTool(),
}


def get_tool(name: str) -> Optional[BaseTool]:
    """Return a tool instance by name."""
    return TOOL_INSTANCES.get(name)
