import time
import re
import base64
import io
import asyncio
import torch
from PIL import Image
from typing import Dict, Optional, List, Any
from sentence_transformers import util
from src.engines.guardrail.repositories import MLRepository, CacheRepository, LogRepository
from src.engines.guardrail.schemas import GuardrailRequestDTO, GuardrailResponseDTO, GuardrailStatus
from src.engines.guardrail.models import GuardrailLog

class TextGuardrailService:
    def __init__(self, ml_repo: MLRepository):
        self.ml_repo = ml_repo
        self.injection_patterns = [r"ignore previous instructions", r"system prompt", r"developer mode"]
        self.policy_denylist = ["nude", "violence", "hate", "explicit"]
        self.food_categories = ["pizza", "burger", "cake", "sushi", "salad", "pasta", "fruit", "vegetable", "meal", "dish"]
        self.food_embeddings = None
    
    def _precompute_food_embeddings(self):
        """Pre-compute food category embeddings for faster validation."""
        if self.food_embeddings is None:
            model = self.ml_repo.get_text_model()
            self.food_embeddings = model.encode(self.food_categories, convert_to_tensor=True)
    
    async def check_injection(self, prompt: str) -> Dict[str, Any]:
        score = 0.0
        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.5
        return {"injection_score": min(score, 1.0)}
    
    async def check_policy(self, prompt: str) -> Dict[str, Any]:
        score = 0.0
        prompt_lower = prompt.lower()
        for term in self.policy_denylist:
            if term in prompt_lower:
                score += 0.5
        return {"policy_score": min(score, 1.0)}
    
    def _sync_check_food_domain(self, prompt: str) -> Dict[str, Any]:
        """Synchronous food domain check - to be run in thread pool."""
        model = self.ml_repo.get_text_model()
        prompt_emb = model.encode(prompt, convert_to_tensor=True)
        if self.food_embeddings is None:
            self.food_embeddings = model.encode(self.food_categories, convert_to_tensor=True)
        similarities = util.cos_sim(prompt_emb, self.food_embeddings)
        max_similarity = float(similarities.max())
        return {"food_domain_score": max_similarity, "is_food_related": max_similarity > 0.35}
    
    async def check_food_domain(self, prompt: str) -> Dict[str, Any]:
        """Check if prompt is food-related using sentence embeddings.
        
        Runs ML inference in thread pool to avoid blocking the event loop.
        """
        return await asyncio.to_thread(self._sync_check_food_domain, prompt)

class ImageGuardrailService:
    def __init__(self, ml_repo: MLRepository):
        self.ml_repo = ml_repo
        # More descriptive labels for better contrast
        self.labels = [
            "a photo of food, a dish, or a meal",           # Food
            "a photo of a person or human face",            # Person
            "a photo of an animal, bird, or insect",        # Animal (e.g. Peacock)
            "a photo of nature, plants, or landscape",      # Nature
            "a photo of an object, tool, or technology",    # Object
            "an explicit, nsfw, or suggestive photo"        # NSFW
        ]
    
    def _sync_check_food_nsfw_clip(self, image_base64: str) -> Dict[str, Any]:
        """Synchronous CLIP check - to be run in thread pool."""
        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Get model and preprocess
            model, preprocess = self.ml_repo.get_clip_model()
            import open_clip
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Prepare inputs
            image_input = preprocess(image).unsqueeze(0).to(self.ml_repo.device)
            text_input = tokenizer(self.labels).to(self.ml_repo.device)
            
            # Inference
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = probs.cpu().numpy()[0]
            
            # Logic: Food score must be the highest AND above a threshold
            food_score = float(probs[0])
            nsfw_score = float(probs[5])
            
            # Find the index of the highest probability
            max_idx = probs.argmax()
            
            return {
                "food_score": food_score,
                "nsfw_score": nsfw_score,
                "top_category": self.labels[max_idx],
                "is_food": max_idx == 0 and food_score > 0.35, # Must be top category and > 35%
                "is_nsfw": nsfw_score > 0.15
            }
        except Exception as e:
            return {"error": str(e), "food_score": 0.0, "nsfw_score": 0.0}
    
    async def check_food_nsfw_clip(self, image_base64: str) -> Dict[str, Any]:
        """Check image for food/NSFW content using CLIP.
        
        Runs ML inference in thread pool to avoid blocking the event loop.
        """
        return await asyncio.to_thread(self._sync_check_food_nsfw_clip, image_base64)

class GuardrailService:
    def __init__(self, ml_repo, cache_repo, log_repo, text_service, image_service):
        self.ml_repo = ml_repo
        self.cache_repo = cache_repo
        self.log_repo = log_repo
        self.text_service = text_service
        self.image_service = image_service
    
    async def validate(self, request: GuardrailRequestDTO) -> GuardrailResponseDTO:
        start_time = time.time()
        
        # 1. Cache Check
        input_hash = await self.cache_repo.compute_hash(request.prompt, request.image_bytes)
        cached = await self.cache_repo.get(input_hash)
        if cached:
            res = GuardrailResponseDTO(**cached)
            res.metadata["cache_hit"] = True
            return res
            
        all_scores = {}
        reasons = []
        
        # 2. Text Validation
        injection = await self.text_service.check_injection(request.prompt)
        all_scores.update(injection)
        if injection["injection_score"] >= 0.5:
            reasons.append("Prompt injection detected")
            
        policy = await self.text_service.check_policy(request.prompt)
        all_scores.update(policy)
        if policy["policy_score"] >= 0.5:
            reasons.append("Policy violation in text")
            
        food_text = await self.text_service.check_food_domain(request.prompt)
        all_scores.update(food_text)
        if not food_text["is_food_related"]:
            reasons.append("Prompt is not food-related")
            
        # 3. Image Validation (if present)
        if request.image_bytes:
            image_res = await self.image_service.check_food_nsfw_clip(request.image_bytes)
            all_scores.update(image_res)
            if image_res.get("is_nsfw"):
                reasons.append("NSFW content detected in image")
            if not image_res.get("is_food") and not image_res.get("error"):
                reasons.append("Image is not food-related")
        
        # 4. Final Decision
        status = GuardrailStatus.BLOCK if reasons else GuardrailStatus.PASS
        processing_time = int((time.time() - start_time) * 1000)
        
        res_data = {
            "status": status,
            "reasons": reasons,
            "scores": {k: v for k, v in all_scores.items() if isinstance(v, (float, int))},
            "metadata": {
                "processing_time_ms": processing_time,
                "cache_hit": False
            }
        }
        
        # 5. Logging & Caching
        await self.log_repo.save(GuardrailLog(
            prompt=request.prompt,
            status=status.value,
            reasons=", ".join(reasons) if reasons else None,
            processing_time_ms=processing_time
        ))
        await self.cache_repo.set(input_hash, res_data)
        
        return GuardrailResponseDTO(**res_data)
