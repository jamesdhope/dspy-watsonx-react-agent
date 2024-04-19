from pymilvus import DataType, Milvus, IndexType, FieldSchema, CollectionSchema,Collection, connections
import requests
from bs4 import BeautifulSoup
import numpy as np
from embedding import EmbeddingFunction

connections.connect(
  alias="default",
  uri="http://localhost:19530",
  token="root:Milvus",
)

# Fetch data from Wikipedia
url = 'https://en.wikipedia.org/wiki/Main_Page'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract article titles and content
article_titles = []
article_contents = []

# Get a list of all Wikipedia article titles
url = "https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch=turing"
response = requests.get(url)
data = response.json()

# Extract the article titles and contents
article_titles = []
article_contents = []
for page in data['query']['search']:
    title = page['title']
    article_titles.append(title)
    
    article_url = f"https://en.wikipedia.org/wiki/{title}"
    article_response = requests.get(article_url)
    soup = BeautifulSoup(article_response.content, 'html.parser')
    
    # Extract the article content
    content = soup.find('div', {'class': 'mw-parser-output'}).get_text()
    article_contents.append(content[:1000])

print(len(article_contents))

# Create a Milvus collection
collection_name = 'wikipedia_articles'
dimension = 768  # Assuming you're using a pre-trained text embedding model
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR,max_length=65535),
    FieldSchema(name='text', dtype=DataType.VARCHAR,max_length=65535),
    FieldSchema(name='long_text', dtype=DataType.VARCHAR,max_length=65535),
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=dimension)
]

schema = CollectionSchema(fields=fields)

collection = Collection(collection_name,schema)

#print(article_contents)

entities = [{'title': title, 'text': text, 'long_text': long_text, 'vector': embedding} for title, text, long_text, embedding in zip(article_titles, article_contents,article_contents, EmbeddingFunction().__call__(article_contents))]

#print(entities)

collection.insert(entities)

index_param = {
    "index_type": IndexType.IVF_SQ8,
    "metric_type": "L2",
    "params": {
        "nlist": 16384
    }
}

collection.create_index(field_name="vector", index_params=index_param)