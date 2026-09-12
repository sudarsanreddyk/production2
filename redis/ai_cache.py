import redis
import time

# Connect to Redis
cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def simulate_llm_call(prompt):
    """Simulates an expensive API call to an LLM like OpenAI or Anthropic."""
    print("🧠 Thinking... (Calling LLM API)")
    time.sleep(3) # Simulating a 3-second delay
    return f"This is the AI's generated response to: '{prompt}'"

def get_response(prompt):
    """Checks cache first; if empty, calls LLM and saves to cache."""
    # Step 1: Check Redis
    cached_answer = cache.get(prompt)
    
    # Step 2: Cache Hit
    if cached_answer:
        print("⚡ CACHE HIT! Returning instantly.")
        return cached_answer
        
    # Step 3: Cache Miss
    print("🐢 CACHE MISS! Need to generate it.")
    new_answer = simulate_llm_call(prompt)
    
    # Step 4: Store in Redis for next time
    cache.set(prompt, new_answer, ex=60) 
    # ex=60 sec this ttl(time to live), after this time, the prompt will be deleted.
    
    # Step 5: Return
    return new_answer

# Let's test it!
user_question = "What is the capital of India?"

print("--- First Request ---")
start_time = time.time()
print(get_response(user_question))
print(f"Time taken: {time.time() - start_time:.2f} seconds\n")

print("--- Second Request (Same Question) ---")
start_time = time.time()
print(get_response(user_question))
print(f"Time taken: {time.time() - start_time:.4f} seconds")