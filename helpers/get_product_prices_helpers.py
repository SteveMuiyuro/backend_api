import json
import redis
import os
# redis_client = redis.Redis(host='localhost', port=6379, db=0) #Local
redis_client = redis.Redis.from_url(os.environ.get('REDIS_URL')) #Production



# Helper functions for Redis session management
def set_session_data(user_id, data):
    redis_client.setex(f"session:{user_id}", 3600, json.dumps(data))  # Session expires in 1 hour

def get_session_data(user_id):
    session_data = redis_client.get(f"session:{user_id}")
    return json.loads(session_data) if session_data else None

def delete_session_data(user_id):
    redis_client.delete(f"session:{user_id}")
