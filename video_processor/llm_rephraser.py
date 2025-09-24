"""
LLM-based text rephraser using OpenAI API
"""
import openai
import os
from typing import List, Optional
import logging
import time
from .speech_to_text import SpeechSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMRephraser:
    """Rephrase text using Large Language Models while maintaining meaning"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM rephraser
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use for rephrasing
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # System prompt for rephrasing
        self.system_prompt = """You are a professional text rephraser. Your task is to rephrase the given text while:

1. Maintaining the exact same meaning and intent
2. Keeping similar length (within 20% of original)
3. Using natural, conversational language
4. Preserving the emotional tone
5. Making it sound more engaging and natural for speech

Please provide only the rephrased text without any explanations or additional comments."""
    
    def rephrase_text(self, text: str, max_retries: int = 3) -> str:
        """
        Rephrase a single text segment
        
        Args:
            text: Original text to rephrase
            max_retries: Number of retry attempts for API calls
            
        Returns:
            Rephrased text
        """
        if not text.strip():
            return text
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Rephrasing (attempt {attempt + 1}): {text[:50]}...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Rephrase this text: {text}"}
                    ],
                    max_tokens=min(len(text.split()) * 3, 500),  # Adaptive token limit
                    temperature=0.7,
                    top_p=0.9
                )
                
                rephrased = response.choices[0].message.content.strip()
                
                # Basic validation
                if rephrased and len(rephrased) > 0:
                    logger.debug(f"Rephrased: {rephrased[:50]}...")
                    return rephrased
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        # If all attempts failed, return original text
        logger.error(f"Failed to rephrase text after {max_retries} attempts, using original")
        return text
    
    def rephrase_segments(self, segments: List[SpeechSegment]) -> List[SpeechSegment]:
        """
        Rephrase all speech segments
        
        Args:
            segments: List of original speech segments
            
        Returns:
            List of speech segments with rephrased text
        """
        rephrased_segments = []
        
        logger.info(f"Rephrasing {len(segments)} segments...")
        
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i + 1}/{len(segments)}")
            
            try:
                rephrased_text = self.rephrase_text(segment.text)
                
                # Create new segment with rephrased text but same timing
                rephrased_segment = SpeechSegment(
                    text=rephrased_text,
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.confidence
                )
                
                rephrased_segments.append(rephrased_segment)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to rephrase segment {i + 1}: {e}")
                # Use original segment if rephrasing fails
                rephrased_segments.append(segment)
        
        logger.info("Rephrasing completed")
        return rephrased_segments
    
    def batch_rephrase(self, segments: List[SpeechSegment], 
                      batch_size: int = 5) -> List[SpeechSegment]:
        """
        Rephrase segments in batches for efficiency
        
        Args:
            segments: List of speech segments
            batch_size: Number of segments to process in each batch
            
        Returns:
            List of rephrased segments
        """
        rephrased_segments = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_texts = [seg.text for seg in batch]
            
            try:
                # Create batch prompt
                batch_prompt = "Rephrase each of the following texts (separated by '---'):\n\n"
                batch_prompt += "\n---\n".join(batch_texts)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt + "\nReturn the rephrased texts separated by '---' in the same order."},
                        {"role": "user", "content": batch_prompt}
                    ],
                    max_tokens=sum(len(text.split()) * 3 for text in batch_texts),
                    temperature=0.7
                )
                
                rephrased_batch = response.choices[0].message.content.strip().split("---")
                
                # Create rephrased segments
                for j, (original_segment, rephrased_text) in enumerate(zip(batch, rephrased_batch)):
                    if j < len(rephrased_batch) and rephrased_text.strip():
                        rephrased_segment = SpeechSegment(
                            text=rephrased_text.strip(),
                            start=original_segment.start,
                            end=original_segment.end,
                            confidence=original_segment.confidence
                        )
                        rephrased_segments.append(rephrased_segment)
                    else:
                        # Fallback to original if batch parsing fails
                        rephrased_segments.append(original_segment)
                
            except Exception as e:
                logger.error(f"Batch rephrasing failed: {e}")
                # Fallback to individual rephrasing
                for segment in batch:
                    rephrased_text = self.rephrase_text(segment.text)
                    rephrased_segment = SpeechSegment(
                        text=rephrased_text,
                        start=segment.start,
                        end=segment.end,
                        confidence=segment.confidence
                    )
                    rephrased_segments.append(rephrased_segment)
            
            # Add delay between batches
            time.sleep(1)
        
        return rephrased_segments