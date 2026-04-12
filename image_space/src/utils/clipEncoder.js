/**
 * clipEncoder.js — CLIP text encoding for ImageSpace search
 * 
 * Uses Transformers.js (ONNX runtime in browser)
 * Loads CLIP model on first search, caches for subsequent queries
 */

import { AutoTokenizer, AutoModel, env } from '@xenova/transformers';

// Configure Transformers.js to use local models if available
env.cacheDir = './models';
env.allowLocalModels = false; // download from Hugging Face

class CLIPEncoder {
  constructor() {
    this.tokenizer = null;
    this.model = null;
    this.isLoading = false;
    this.loadPromise = null;
  }

  /**
   * Load CLIP model and tokenizer
   * This is lazy-loaded on first use
   */
  async load() {
    if (this.model && this.tokenizer) return;
    if (this.isLoading) {
      await this.loadPromise;
      return;
    }

    this.isLoading = true;
    this.loadPromise = this._doLoad();
    await this.loadPromise;
  }

  async _doLoad() {
    // Using Xenova's CLIP-ViT-B-32 model (ONNX quantized)
    // This is ~150MB download on first use, cached thereafter
    const modelId = 'Xenova/clip-vit-base-patch32';
    
    this.tokenizer = await AutoTokenizer.from_pretrained(modelId);
    this.model = await AutoModel.from_pretrained(modelId, {
      quantized: true, // Use INT8 quantized model (~63MB vs ~250MB)
    });
    
    this.isLoading = false;
    console.log('[CLIP] Model loaded');
  }

  /**
   * Encode text to 512D CLIP embedding
   * Returns Float32Array(512) normalized
   */
  async encodeText(text) {
    await this.load();

    // Tokenize
    const inputs = this.tokenizer(text, {
      truncation: true,
      max_length: 77,
      padding: 'max_length',
      return_tensors: 'pt',
    });

    // Run inference
    const outputs = await this.model(inputs);
    
    // Get text embedding (pooler_output or last hidden state mean)
    const embedding = outputs.text_embeds.data;
    
    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
    const normalized = embedding.map(v => v / (norm || 1));
    
    return new Float32Array(normalized);
  }

  /**
   * Check if model is ready for encoding
   */
  get isReady() {
    return this.model !== null && this.tokenizer !== null;
  }
}

// Singleton instance
const clipEncoder = new CLIPEncoder();

export async function encodeSearchQuery(text) {
  if (!text || text.trim().length === 0) {
    return null;
  }
  
  try {
    const embedding = await clipEncoder.encodeText(text.trim());
    return embedding;
  } catch (err) {
    console.error('[CLIP] Encoding failed:', err);
    return null;
  }
}

export function getClipEncoderStatus() {
  return {
    isReady: clipEncoder.isReady,
    isLoading: clipEncoder.isLoading,
  };
}

export { clipEncoder };
