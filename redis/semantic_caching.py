"""
if User A asks "What is the capital of France?" 
and User B asks "Can you tell me the capital of France?" 
— will User B get our fast cached answer, 
or the slow 3-second delay?
A: the slow 3-second delay because it checks exact same text.
So, the solution is semantic caching
"""
"""
How can we do semantic caching?
Embeddings because redis supports vector search
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_URL"),  
    api_key=os.getenv("api_key")     
)

def get_embedding(text:str):
    response=client.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_MODEL","nomic-embed-text-v2-moe-GGUF")
    )
    return response.data[0].embedding

my_prompt= "what is the capital of India?"
prompt_vector=get_embedding(my_prompt)

print(f"Success! Converted our prompt into a list of {len(prompt_vector)} numbers.")

"""
We need to create a vector index.
To do this, we need to tell Redis exactly what shape our data is
in and how to measure the "distance" between two prompts to see
if they mean the same thing.
"""
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
import numpy as np

# 1. Define the blueprint (schema) for our cache
schema=(
    TextField("prompt"),
    TextField("response"),
    VectorField("embedding", "FLAT", #Use "HNSW" for DIM=768+ to get O(log N) search speed.
                {
                    "Type": "FLOAT32",
                    "DIM" : len(prompt_vector),
                    "DISTANCE_METRIC" : "COSINE"
                }
            )
)

# Connect to Redis
cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 2. Create the index in Redis (we use try/except in case we run it twice)
try:
    cache.ft("idx:semantic_cache").create_index(
        schema,
        definition=IndexDefinition(prefix=["cache:"],index_type=IndexType.HASH)
    )
    print("Created Vector Index in Redis!")

except Exception as e:
    print("Index already exists.")

# 3. Store our data! (Redis requires vectors to be in 'bytes' format)
vector_bytes = np.array(prompt_vector, dtype=np.float32).tobytes()

cache.hset("cache:1", mapping={
    "prompt" : my_prompt,
    "response" : "Delhi is the capital of India.",
    "embedding": vector_bytes
})

print("Successfully saved our first semantic cache entry!")

"""
USers question
"""

from redis.commands.search.query import Query

def check_semantic_cache(new_question):
    
    # 1. Convert User B's question into an embedding
    query_vector = get_embedding(new_question)
    vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

    # 2. Build the Redis Vector Search Query (KNN 1 = find the 1 closest match)
    q = Query("*=>[KNN 1 @embedding $query_vec AS vector_score]")\
        .return_fields("prompt", "response", "vector_score")\
        .sort_by("vector_score")\
        .dialect(2)

    # 3. Execute the search
    results = cache.ft("idx:semantic_cache").search(
        q,
        query_params={"query_vec": vector_bytes}
    )
    
    return results

# Let's test it with User B's slightly different question!
user_b_question = "Can you tell me the capital of India?"
print(f"\nUser B asked: '{user_b_question}'")
results = check_semantic_cache(user_b_question)

if results.total > 0:
    match = results.docs[0]
    print(f"Matched Cached Prompt: '{match.prompt}'")
    print(f"Cached Response: '{match.response}'")
    print(f"Distance Score: {match.vector_score}")
else:
    print("No matches found.")

"""
Now, imagine User C comes along and asks a completely different
question: "What is the capital of Japan?"

If we convert that to a vector and compare it to our cached 
question ("What is the capital of India?"), the distance score 
will be much higher—maybe something like 0.45 or 0.60, 
because the underlying meanings are different.
Solution: Use threshold
"""
user_c_question = "Can you tell me the capital of Japan?"
print(f"\nUser c asked: '{user_c_question}'")
results = check_semantic_cache(user_c_question)
threshold=0.15
if results.total > 0:
    match = results.docs[0]
    print(f"Full Document Object: {match}")
    print(f"Vector Distance Score: {match.vector_score}")
    
    # FIX 1: Use bracket notation [] instead of .get()
    cached_prompt = match['prompt']
    cached_response = match['response']
    
    # FIX 2: Convert vector_score from string to float for comparison
    distance = float(match.vector_score)
    
    if distance < threshold:
        print(f"Matched Cached Prompt: '{cached_prompt}'")
        print(f"Cached Response: '{cached_response}'")
        print(f"Distance Score: {distance:.4f} (below threshold {threshold})")
    else:
        print(f"Result found but distance ({distance:.4f}) exceeds threshold ({threshold}).")

