"""
MYND Brain - Knowledge Extractor
=================================
Extracts meaningful concepts from conversations and integrates them into the map.

Conversations → Concepts → Map Nodes

Only extracts MEANINGFUL, REUSABLE knowledge.
Skips small talk, debugging back-and-forth, etc.
"""

import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .map_vector_db import MapVectorDB, UnifiedNode, SourceRef
from .conversation_archive import ArchivedConversation


@dataclass
class ExtractedConcept:
    """A concept extracted from a conversation."""
    label: str
    description: str
    concept_type: str  # 'insight', 'decision', 'technique', 'fact', 'goal', 'question'
    context: str  # Relevant excerpt from conversation
    confidence: float  # 0-1


EXTRACTION_PROMPT = """Analyze this conversation and extract the key concepts, insights, and knowledge.

For each meaningful item, provide:
- label: Short name (2-5 words)
- description: One sentence explanation
- type: One of: insight, decision, technique, fact, goal, question
- context: The relevant excerpt (1-2 sentences max)
- confidence: 0.0 to 1.0 (how clearly this was stated)

RULES:
- Only extract MEANINGFUL, REUSABLE knowledge
- Skip greetings, small talk, "let me help you", etc.
- Skip debugging back-and-forth and error messages
- Skip meta-discussion about the conversation itself
- Merge similar ideas into one concept
- Be specific: "Time blocking technique" not just "Productivity"
- Maximum 10 concepts per conversation

CONVERSATION:
---
{conversation}
---

Return ONLY a JSON array of concepts. Example:
[
  {{"label": "Eisenhower Matrix", "description": "Prioritization framework using urgent/important axes", "type": "technique", "context": "The Eisenhower Matrix helps decide what to do first", "confidence": 0.9}},
  {{"label": "Morning routine importance", "description": "Starting the day with intention improves focus", "type": "insight", "context": "I've found that having a morning routine...", "confidence": 0.7}}
]

JSON array:"""


class KnowledgeExtractor:
    """
    Extracts knowledge from conversations and integrates into the map.
    """

    def __init__(
        self,
        map_db: MapVectorDB,
        api_key: Optional[str] = None,
        model: str = 'claude-3-haiku-20240307'
    ):
        """
        Initialize the knowledge extractor.

        Args:
            map_db: The unified map vector database
            api_key: Anthropic API key (optional, uses rule-based if not provided)
            model: Claude model to use for extraction
        """
        self.map_db = map_db
        self.api_key = api_key
        self.model = model

        # Stats
        self.total_extracted = 0
        self.total_integrated = 0

    async def extract_concepts(
        self,
        conversation: ArchivedConversation,
        use_ai: bool = True
    ) -> List[ExtractedConcept]:
        """
        Extract concepts from a conversation.

        Args:
            conversation: The conversation to process
            use_ai: Whether to use AI extraction (vs rule-based)

        Returns:
            List of extracted concepts
        """
        if use_ai and self.api_key:
            return await self._extract_with_ai(conversation)
        else:
            return self._extract_rule_based(conversation)

    async def _extract_with_ai(
        self,
        conversation: ArchivedConversation
    ) -> List[ExtractedConcept]:
        """Extract concepts using Claude API."""
        try:
            import httpx

            # Truncate conversation if too long
            text = conversation.text
            if len(text) > 30000:
                # Take beginning and end
                text = text[:15000] + "\n\n...[truncated]...\n\n" + text[-15000:]

            prompt = EXTRACTION_PROMPT.format(conversation=text)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://api.anthropic.com/v1/messages',
                    headers={
                        'x-api-key': self.api_key,
                        'anthropic-version': '2023-06-01',
                        'content-type': 'application/json'
                    },
                    json={
                        'model': self.model,
                        'max_tokens': 2000,
                        'messages': [{'role': 'user', 'content': prompt}]
                    },
                    timeout=60.0
                )

                if response.status_code != 200:
                    print(f"⚠️ AI extraction failed: {response.status_code}")
                    return self._extract_rule_based(conversation)

                data = response.json()
                content = data['content'][0]['text']

                # Parse JSON response
                concepts = self._parse_concepts_json(content)
                print(f"🧠 AI extracted {len(concepts)} concepts")
                return concepts

        except Exception as e:
            print(f"⚠️ AI extraction error: {e}")
            return self._extract_rule_based(conversation)

    def _parse_concepts_json(self, content: str) -> List[ExtractedConcept]:
        """Parse the JSON response from AI."""
        concepts = []

        try:
            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', content)
            if not match:
                return concepts

            data = json.loads(match.group())

            for item in data:
                if not isinstance(item, dict):
                    continue

                concepts.append(ExtractedConcept(
                    label=item.get('label', '')[:100],
                    description=item.get('description', '')[:500],
                    concept_type=item.get('type', 'insight'),
                    context=item.get('context', '')[:500],
                    confidence=float(item.get('confidence', 0.5))
                ))

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")

        return concepts

    def _extract_rule_based(
        self,
        conversation: ArchivedConversation
    ) -> List[ExtractedConcept]:
        """
        Simple rule-based extraction as fallback.
        Extracts based on patterns and keywords.
        """
        concepts = []
        text = conversation.text

        # Pattern: "The key insight is..." or "Important: ..."
        insight_patterns = [
            r'(?:key insight|important point|main takeaway|crucial|essential)[:\s]+([^.!?]+[.!?])',
            r'(?:I learned|we discovered|turns out)[:\s]+([^.!?]+[.!?])',
            r'(?:The solution|The answer|The approach)[:\s]+([^.!?]+[.!?])',
        ]

        for pattern in insight_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:  # Max 2 per pattern
                if len(match) > 20:  # Skip too short
                    concepts.append(ExtractedConcept(
                        label=match[:50].strip(),
                        description=match.strip(),
                        concept_type='insight',
                        context=match[:200],
                        confidence=0.5
                    ))

        # Pattern: Decisions made
        decision_patterns = [
            r'(?:decided to|going to|will|should)[:\s]+([^.!?]+[.!?])',
            r'(?:the plan is|next step is)[:\s]+([^.!?]+[.!?])',
        ]

        for pattern in decision_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                if len(match) > 20:
                    concepts.append(ExtractedConcept(
                        label=match[:50].strip(),
                        description=match.strip(),
                        concept_type='decision',
                        context=match[:200],
                        confidence=0.4
                    ))

        # Deduplicate by label similarity
        seen = set()
        unique_concepts = []
        for c in concepts:
            key = c.label.lower()[:30]
            if key not in seen:
                seen.add(key)
                unique_concepts.append(c)

        print(f"📋 Rule-based extracted {len(unique_concepts)} concepts")
        return unique_concepts[:10]  # Max 10

    async def integrate_concept(
        self,
        concept: ExtractedConcept,
        conversation: ArchivedConversation
    ) -> Dict:
        """
        Integrate an extracted concept into the map.

        Args:
            concept: The concept to integrate
            conversation: Source conversation

        Returns:
            Dict with integration result
        """
        # Create source reference
        source_ref = SourceRef(
            conversation_id=conversation.id,
            excerpt=concept.context[:300],
            extracted_at=datetime.utcnow().isoformat()
        )

        # Find or create node
        node, is_new = self.map_db.find_or_create_node(
            label=concept.label,
            description=concept.description,
            similarity_threshold=0.80,  # High threshold to avoid duplicates
            source=source_ref
        )

        # Update node type if it's new
        if is_new:
            node.type = concept.concept_type
            node.confidence = concept.confidence

        self.total_integrated += 1

        return {
            'node_id': node.id,
            'label': node.label,
            'is_new': is_new,
            'action': 'created' if is_new else 'enriched'
        }

    async def process_conversation(
        self,
        conversation: ArchivedConversation,
        use_ai: bool = True
    ) -> Dict:
        """
        Process a conversation: extract concepts and integrate into map.

        Args:
            conversation: The conversation to process
            use_ai: Whether to use AI extraction

        Returns:
            Processing result summary
        """
        start_time = datetime.utcnow()

        # Extract concepts
        concepts = await self.extract_concepts(conversation, use_ai=use_ai)
        self.total_extracted += len(concepts)

        # Integrate each concept
        results = []
        node_ids = []

        for concept in concepts:
            if concept.confidence < 0.3:  # Skip low confidence
                continue

            result = await self.integrate_concept(concept, conversation)
            results.append(result)
            node_ids.append(result['node_id'])

        # Save map
        self.map_db.save()

        # Summary
        created = sum(1 for r in results if r['is_new'])
        enriched = len(results) - created

        return {
            'conversation_id': conversation.id,
            'conversation_title': conversation.title,
            'concepts_extracted': len(concepts),
            'nodes_created': created,
            'nodes_enriched': enriched,
            'node_ids': node_ids,
            'processing_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
        }

    async def process_batch(
        self,
        conversations: List[ArchivedConversation],
        use_ai: bool = True
    ) -> List[Dict]:
        """Process multiple conversations."""
        results = []

        for conv in conversations:
            result = await self.process_conversation(conv, use_ai=use_ai)
            results.append(result)
            print(f"✅ Processed: {conv.title[:50]}... → {result['nodes_created']} new, {result['nodes_enriched']} enriched")

        return results

    def get_stats(self) -> Dict:
        """Get extractor statistics."""
        return {
            'total_extracted': self.total_extracted,
            'total_integrated': self.total_integrated,
            'map_stats': self.map_db.get_stats()
        }
