from vectorstore.qdrant_client import get_qdrant_client


client = get_qdrant_client()

print("Connecting to Qdrant...")

collections = client.get_collections()

print("Connection successful!")
print("Existing collections:")

for collection in collections.collections:
    print("-", collection.name)