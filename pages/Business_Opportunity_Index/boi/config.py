# ---------- Business rules ----------
CITIES = {
    "Lagos":      {"lat": 6.465,  "lon": 3.406,  "iso3": "NGA"},
    "Medellín":   {"lat": 6.251,  "lon": -75.56, "iso3": "COL"},
    "Phnom Penh": {"lat": 11.556, "lon": 104.9,  "iso3": "KHM"},
    "Da Nang":    {"lat": 16.054, "lon": 108.2,  "iso3": "VNM"},
    "Surat":      {"lat": 21.17,  "lon": 72.83,  "iso3": "IND"}
}

POI_CATEGORIES = {
    "laundromat": {"key": "amenity", "value": "laundry"},
    "clinic":     {"key": "amenity", "value": "clinic"},
}

BENCHMARK = {          # businesses per 10 k residents
    "laundromat": 0.8,
    "clinic":     0.25,
}

# --- BigQuery ---
PROJECT   = "mindful-vial-460001-h6"
DATASET   = "boi_mvp"          # created once via infra/bigquery.sql
LOCATION  = "us-central1"

# --- Confluent ---
KAFKA_BOOTSTRAP = "pkc-zgp5j7.us-south1.gcp.confluent.cloud:9092"
KAFKA_API_KEY   = "LKH74XFTETPCQOPN"
KAFKA_SECRET    = "qUt3+K7gyZ296T2rUph0Y6fTWaDOEgzEFqWqJw6Sa1Z6ris6TfU+9gRjzEFdqi64"
# Toggle streaming vs. direct-to-BigQuery
USE_KAFKA = False      # False = run_pipeline --mode direct / local
                       # True  = run_pipeline --mode stream
TOPIC_DEMAND    = "boi.demand"
TOPIC_SUPPLY    = "boi.supply"

# ── Foursquare Places (free dev tier: 950 calls/day) ──
FSQ_API_KEY = "fsq3oRO8wX0LN4zbOfOOCRhzvN8elEbTW131PcGW47Nkiho="
# developer.foursquare.com → “Create a new App” → Generate Key

