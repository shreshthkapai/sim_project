"""
injection.py

Scenario injection (external input):
- Parse natural language query via LLM
- Extract entities + categories
- Map to graph nodes
- Set external input s_i(t)
"""

import requests
import json
import time
import math
from typing import Dict, List, Tuple, Optional

from sim_config import ENTITY_PREFIX


# LLM Configuration
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Category definitions
CATEGORY_DEFINITIONS = {
    'Travel': 'trips, vacations, tourism, destinations, hotels, flights',
    'Work_Career': 'jobs, employment, career development, workplace, skills',
    'Daily_Life': 'routine activities, errands, household tasks, local services',
    'Life_Transitions': 'major life changes, moving, relationships, births, deaths',
    'Location': 'maps, directions, addresses, geographic information, places',
    'Entertainment': 'movies, music, games, hobbies, leisure activities',
    'Technology': 'gadgets, software, apps, electronics, tech products',
    'Fashion': 'clothing, style, shopping, fashion trends, apparel',
    'News_Politics': 'current events, politics, news articles, elections'
}


class QueryInjector:
    """Handles query parsing and injection into simulation state."""
    
    def __init__(self, graph_loader):
        """
        Initialize injector.
        
        Args:
            graph_loader: FrozenGraphLoader instance
        """
        self.graph_loader = graph_loader
        self.kg = graph_loader.kg  # Access to knowledge graph for mention_count
    
    def inject_query(self, query: str, state, 
                default_strength: float = 1.0,
                use_importance_weighting: bool = True) -> Dict:
        """
        Parse query and inject external input into simulation state.
        
        Args:
            query: Natural language query
            state: SimulationState instance
            default_strength: Base injection strength
            use_importance_weighting: Weight by entity mention_count
            
        Returns:
            Dict with injection statistics
        """
        print(f"\n{'='*60}")
        print(f"INJECTING QUERY: '{query}'")
        print(f"{'='*60}")
        
        # 1. Extract entities + categories via LLM
        extraction = self._extract_entities_categories(query)
        
        if not extraction:
            print("⚠️  LLM extraction failed - no injection performed")
            return {'entities_injected': 0, 'categories_injected': 0}
        
        entities = extraction.get('entities', [])
        categories = extraction.get('categories', {})
        
        print(f"\nLLM Extraction:")
        print(f"  Entities: {entities}")
        print(f"  Categories: {categories}")
        
        # 2. Inject entities and categories
        entities_injected = 0
        categories_injected = 0
        injection_details = []
        
        # Inject entities ONLY IF THEY EXIST (no graph modification!)
        for entity_name in entities:
            entity_id = self._normalize_entity_name(entity_name)
            
            # CRITICAL: Only inject if entity exists in frozen graph
            if entity_id in self.graph_loader.node_to_idx:
                idx = self.graph_loader.node_to_idx[entity_id]
                
                # Calculate injection strength
                if use_importance_weighting:
                    strength = self._calculate_entity_strength(entity_id, default_strength)
                else:
                    strength = default_strength
                
                state.s[idx] = strength
                entities_injected += 1
                injection_details.append({
                    'type': 'entity',
                    'name': entity_name,
                    'id': entity_id,
                    'strength': strength
                })
                print(f"  ✓ Injected entity: {entity_name} (strength={strength:.3f})")
            else:
                # Entity doesn't exist - this is NORMAL and OK
                # The category injection will handle the semantic meaning
                print(f"  ○ Entity not in graph: {entity_name} (semantic captured by categories)")
        
        # Inject categories (always inject, weighted by LLM confidence)
        for category, confidence in categories.items():
            if category in self.graph_loader.node_to_idx:
                idx = self.graph_loader.node_to_idx[category]
                strength = confidence * default_strength
                state.s[idx] = strength
                categories_injected += 1
                injection_details.append({
                    'type': 'category',
                    'name': category,
                    'strength': strength
                })
                print(f"  ✓ Injected category: {category} (strength={strength:.3f})")
        
        print(f"\nInjection Summary:")
        print(f"  Entities injected: {entities_injected}/{len(entities)}")
        print(f"  Categories injected: {categories_injected}/{len(categories)}")
        print(f"{'='*60}\n")
        
        return {
            'entities_injected': entities_injected,
            'categories_injected': categories_injected,
            'total_injected': entities_injected + categories_injected,
            'details': injection_details,
            'extraction': extraction
        }
    
    def _extract_entities_categories(self, query: str) -> Optional[Dict]:
        """
        Extract entities and categories from query using LLM.
        
        Returns:
            Dict with 'entities' (list) and 'categories' (dict)
        """
        # Build category descriptions
        cat_descriptions = "\n".join([
            f"- {cat}: {desc}" 
            for cat, desc in CATEGORY_DEFINITIONS.items()
        ])
        
        prompt = f"""Analyze this search query and extract entities and categories.

EXTRACTION RULES:
1. EXTRACT meaningful concepts. An entity can be multiple words if it represents ONE specific thing.
2. SPLIT distinct ideas. Do not lump "who", "what", and "where" into one string.
3. KEYWORDS: Extract descriptive topics or intent not in the category list.

CATEGORIES (use ONLY these):
{cat_descriptions}

QUERY: "{query}"

OUTPUT FORMAT (JSON only, no explanation):
{{
  "entities": ["Entity One", "Entity Two"],
  "categories": {{"CategoryName": 0.8, "AnotherCategory": 0.5}},
  "attributes": {{}}
}}"""
        
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 300
            }
        }
        
        # Retry logic
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(OLLAMA_API, json=payload, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                result = response.json()
                response_text = result.get('response', '')
                
                # Parse JSON response
                parsed = self._parse_llm_response(response_text)
                
                if parsed:
                    return parsed
                
            except Exception as e:
                print(f"  LLM attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
        
        return None
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse LLM JSON response."""
        try:
            # Try direct parse
            data = json.loads(response)
            return self._validate_extraction(data)
        except:
            pass
        
        try:
            # Try extracting JSON from markdown
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                return None
            
            data = json.loads(json_str)
            return self._validate_extraction(data)
        except:
            return None
    
    def _validate_extraction(self, data: Dict) -> Optional[Dict]:
        """Validate and clean LLM extraction."""
        if not isinstance(data, dict):
            return None
        
        # Extract entities (list of strings)
        entities = data.get('entities', [])
        if not isinstance(entities, list):
            entities = []
        entities = [e for e in entities if isinstance(e, str) and e.strip()]
        
        # Extract categories (dict with float values)
        categories = data.get('categories', {})
        if not isinstance(categories, dict):
            categories = {}
        
        cleaned_categories = {}
        for k, v in categories.items():
            if isinstance(k, str) and v is not None:
                try:
                    cleaned_categories[k] = float(v)
                except (ValueError, TypeError):
                    continue
        
        # Must have at least one category
        if not cleaned_categories:
            return None
        
        return {
            'entities': entities,
            'categories': cleaned_categories,
            'attributes': data.get('attributes', {})
        }
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """
        Normalize entity name to graph node ID format.
        
        Example: "Paris Hotels" -> "entity_paris_hotels"
        """
        normalized = entity_name.lower().replace(' ', '_')
        return f"{ENTITY_PREFIX}{normalized}"
    
    def _calculate_entity_strength(self, entity_id: str, base_strength: float) -> float:
        """
        Calculate injection strength based on entity importance (mention_count).
        Uses same formula as KG training for consistency.
        
        Args:
            entity_id: Entity node ID
            base_strength: Base injection strength
            
        Returns:
            Weighted strength
        """
        # Get entity node from graph
        if entity_id not in self.graph_loader.graph.nodes:
            return base_strength
        
        entity_data = self.graph_loader.graph.nodes[entity_id]
        mention_count = entity_data.get('mention_count', 1)
        
        # Weight by 1.0 + log(1 + mention_count) - SAME AS KG TRAINING
        # mention_count=1 → weight≈1.69, mention_count=10 → weight≈3.4, mention_count=100 → weight≈5.6
        weight_multiplier = 1.0 + math.log(1 + mention_count)
        
        # Apply to base strength
        strength = base_strength * weight_multiplier
        
        return strength