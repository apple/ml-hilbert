#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Semantic Search Engine for Mathematical Theorems and Definitions
Searches through data.jsonl based on informal_description field using semantic similarity
"""

import json
import argparse
import os
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import re
import logging
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    def __init__(self, 
                 cache_dir: str = 'cache/',
                 data_file: str = "data.jsonl", 
                 model_name_or_path: str = "all-mpnet-base-v2",
                 top_k: int = 5):
        
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.index = None
        self.entries = []
        self.cache_dir = cache_dir
        self.data_file = os.path.join(self.cache_dir, data_file)
        self.embeddings_file = os.path.join(self.cache_dir, "embeddings.json")
        self.index_file = os.path.join(self.cache_dir, "faiss_index.bin")
        self.top_k = top_k
        self.initialize()

    def load_model(self):
        """Load the sentence transformer model"""
        logger.info(f"Loading embedding model: {self.model_name_or_path}")
        self.model = SentenceTransformer(self.model_name_or_path, local_files_only=True)
        
    def load_data(self) -> List[Dict]:
        """Load and parse the JSONL data file"""
        logger.info(f"Loading data from {self.data_file}")
        entries = []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Reading JSONL")):
                try:
                    entry = json.loads(line.strip())

                    # Only include entries that have informal_description
                    if entry.get('informal_description'):
                        entries.append({
                            'informal_description': entry['informal_description'],
                            'informal_name': entry.get('informal_name', ''),
                            'module_name': '.'.join(entry.get('module_name', [])),
                            'kind': entry.get('kind', ''),
                            'name': '.'.join(entry.get('name', [])),
                            'line_number': line_num + 1,
                            'signature': entry.get('signature', '')
                        })
                except json.JSONDecodeError as e:
                    logger.info(f"Warning: Skipping malformed JSON on line {line_num + 1}: {e}")
                    continue
                    
        logger.info(f"Loaded {len(entries)} entries with informal descriptions")
        return entries
        
    def generate_embeddings(self, entries: List[Dict]) -> np.ndarray:
        """Generate embeddings for all informal descriptions"""
        logger.info("Generating embeddings for all descriptions...")
        descriptions = [entry['name'] + " : " + entry['informal_name'] + " " + entry['informal_description'] for entry in entries]
        # Generate embeddings in batches for memory efficiency
        batch_size = 64
        embeddings = []
        
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating embeddings"):
            batch = descriptions[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
        
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
    def save_cache(self, entries: List[Dict], embeddings: np.ndarray):
        """Save entries and embeddings to disk for faster subsequent runs"""
        logger.info("Saving cache to disk...")
        
        # Save entries and embeddings
        with open(self.embeddings_file, 'w') as f:
            json.dump({'entries': entries, 'embeddings': embeddings.tolist()}, f)
            
        # Save FAISS index
        faiss.write_index(self.index, self.index_file)
        logger.info("Cache saved successfully")
        
    def load_cache(self) -> Tuple[List[Dict], np.ndarray]:
        """Load cached entries and embeddings from disk"""
        logger.info("Loading cached data...")

        with open(self.embeddings_file, 'r') as f:
            data = json.load(f)
        data['embeddings'] = np.array(data['embeddings'])
            
        self.index = faiss.read_index(self.index_file)
        logger.info(f"Loaded cache with {len(data['entries'])} entries")
        
        return data['embeddings']
        
    def initialize(self):
        """Initialize the search engine - load data and build/load index"""
        self.load_model()
        
        # Check if cache exists
        if os.path.exists(self.embeddings_file) and os.path.exists(self.index_file):
            try:
                embeddings = self.load_cache()
            except Exception as e:
                logger.info(f"Error loading cache: {e}")
                logger.info("Rebuilding from scratch...")
        else:
            embeddings = None
        # Build from scratch
        self.entries = self.load_data()
        
        if embeddings is None:
            embeddings = self.generate_embeddings(self.entries)
            self.build_index(embeddings)
            self.save_cache(self.entries, embeddings)
    
    def _clean_up(self, query: str) -> str:
        """Removes variations of 'lean' and 'lean 4' from the query."""
        # This pattern looks for "lean" (case-insensitive) optionally
        # followed by a space and the number 4.
        # \b ensures we only match whole words.
        pattern = r'\b(lean|Lean)( 4)?\b'
        
        # re.sub replaces all occurrences of the pattern with an empty string.
        cleaned_query = re.sub(pattern, '', query)
        
        # remove any mentions of mathlib
        cleaned_query = cleaned_query.replace('mathlib', '')
        cleaned_query = cleaned_query.replace('Mathlib', '')
        
        # Clean up any extra spaces that might result from the removal.
        return ' '.join(cleaned_query.split())

    def search(self, query: str, top_k: int = None) -> List[Tuple[float, Dict]]:
        """Search for semantically similar entries"""
        if not self.model or not self.index:
            raise ValueError("Search engine not initialized. Call initialize() first.")
        
        # remove irrelevant words from query
        query = self._clean_up(query)
        
        if top_k is None:
            top_k = self.top_k
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search for similar entries
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.entries):  # Valid index
                results.append((float(score), self.entries[idx]))
                
        return results
        
    def get_search_results(self, query: str, top_k: int = 5) -> str:
        """Format search results for display"""
        results = self.search(query, top_k)

        if not results:
            return f"No results found for query: '{query}'"
        return self.format_results(results)
    
    def search_by_name(self, names, exact_match=False):
        """Search for entries by name (exact or partial matching)
        
        Args:
            names: String or list of strings to search for
            exact_match: If True, only return exact matches. If False, return partial matches.
            
        Returns:
            List of matching entries (dictionaries)
        """
        if not self.entries:
            raise ValueError("Search engine not initialized. Call initialize() first.")
        
        # Normalize input to list of strings
        if isinstance(names, str):
            search_terms = [names.lower()]
        else:
            search_terms = [name.lower() for name in names]
        
        results = []
        
        for entry in self.entries:
            # Get searchable fields from entry
            full_name = entry.get('name', '').lower()
            informal_name = entry.get('informal_name', '').lower()
            module_name = entry.get('module_name', '').lower()
            
            # Check if any search term matches this entry
            for search_term in search_terms:
                match_found = False
                
                if exact_match:
                    # Exact matching
                    if (search_term == full_name or 
                        search_term == informal_name or
                        search_term == module_name):
                        match_found = True
                else:
                    # Partial matching
                    if (search_term in full_name or 
                        search_term in informal_name or
                        search_term in module_name):
                        match_found = True
                
                if match_found:
                    results.append((None, entry))
                    break  # Avoid duplicate entries for multiple matching terms
        
        return self.format_results(results)
        
    def format_results(self, results):
        output = []
        for i, (score, entry) in enumerate(results, 1):
            output.append(f"\n{i}. [{entry['kind'].upper()}] {entry['informal_name']}")
            output.append(f"Module: {entry['module_name']}")
            output.append(f"Name: {entry['name']}")
            output.append(f"Description: {entry['informal_description']}")
            if 'signature' in entry:
                output.append(f"Signature: {entry['signature']}")
        return "\n".join(output)

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument('--query', type=str, help='Query to search for')
    args.add_argument('--top_k', type=int, default=5, help='Number of top results to display')
    args = args.parse_args()
    # Initialize the semantic search engine
    search_engine = SemanticSearchEngine(
        cache_dir='cache/',
        data_file='mathlib_informal.jsonl'
    )
    # Perform the search and print the results.
    logger.info(search_engine.get_search_results(args.query))
