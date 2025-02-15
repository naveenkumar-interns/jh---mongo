from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
from datetime import datetime
import pymongo
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List

client = pymongo.MongoClient("mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["jacksonHardwareDB"]
collection = db["inventory"]

app = Flask(__name__)
CORS(app)
# Load environment variables
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

keywords = ""
wordings = ""

memory = ConversationBufferWindowMemory(return_messages=True, k=2)


embedding_cache ={}

def generate_embedding(text: str) -> List[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    print(type(text))
#   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=os.getenv("NOMIC_AI_APIKEY"))
    # embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv("MISTRAL_AI_APIKEY"))
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("hf_token"), model_name="sentence-transformers/all-MiniLM-l6-v2")
    response = embeddings.embed_query(text)
    embedding_cache[text]=response
    return response


def user_intent(text):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """you are helpful assistant that helps to analyze the list of keywords and provide a user intention of which product and its feature he is searching for.
        output should be line of words , that is best for vector search.
        note important: Avoid using technical formatting like new line symbols, markdown symbols *, _, etc., or bullet points.
        """,
    ),
    (
        "human",
        "{input}"
    ),
])
        chain = prompt | llm
        response = chain.invoke({"input": text})
        return response.content
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise


def get_keywords(input_text):
    try:

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """ you are system that helps to extract main keywords from the user query.
        output should be a list of keywords that are extracted from the user query without brackets.
        note important: Avoid using technical formatting like new line symbols, markdown symbols *, _, etc., or bullet points.
        """,
    ),
    (
        "human",
        "{input}"
    ),
])
        chain = prompt | llm
        response = chain.invoke({"input": input_text})
        global wordings
        wordings +=response.content
        if len(wordings) > 100:
            wordings = wordings[20:]

        global keywords
        keywords = user_intent(wordings)
        return keywords
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise

def convert_to_json(data):
    result = []
    forai = []
    for product in data:
        # Filter out unnecessary keys from metadata
        product_info = {
        'id': product.get('id'),
        'title': product.get('title'),
        'description': product.get('description'),
        'product_type': product.get('product_type'),
        'link': product.get('link'),
        'image_list': product.get('image_list'),
        'price': product.get('price'),
        'inventory_quantity': product.get('inventory_quantity'),
        'vendor': product.get('vendor')
        }
        iteminfo = {
        'title': product.get('title'),
        'product_type': product.get('product_type'),
        'description': product.get('description'),
        'vendor': product.get('vendor'),
        'price': product.get('price'),
        'inventory_quantity': product.get('inventory_quantity')
        }
        forai.append(iteminfo)
        result.append(product_info)

    print(result)

    return result,forai


def get_product_search(query):
    results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "embeddings",
        "numCandidates": 100,
        "limit": 8,
        # "index": "vector_search_index",
        "index": "vx",
        }}
    ])
    return convert_to_json(results)


def get_response_product_search(input_text,related_products):
    try:
        prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """You are Jackson Hardware Store's AI assistant. Your job is to:
        1. Help customers find the right tools, hardware, or equipment.
        2. Suggest relevant products based on customer needs and related items.
        3. Share key product details like brand, features, use cases, and availability.
        5. note important: Avoid using technical formatting like new line symbols, markdown symbols *, _, etc., or bullet points.
        6.  Respond in a short, direct manner <maximum 1-2 brief sentences> with only the most relevant information. and max tokens 20

        note: act as a chatbot 
        Deliver the response here in plain text without any formatting.
        chat history: {history}
        """,
    ),
    ("human", "{input}"),
])

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )
        query = f"user query : {input_text} and related products based on user query:{str(related_products)}"
        response = chain.invoke({"input": query})
        return response['text']
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def get_availability(input_text):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """

        You are Jackson Hardware Store's AI assistant. Your job is to:
        
        extract the price and availability of the products asked by the user from the related products list: {input}

        note respond 0$ if not mentioned.

         Respond in a short, direct manner <maximum 1-2 brief sentences> with only the most relevant information. and max tokens 20

        example: user: "I need a hammer." and related products based on user query: <<"title": "Hammer", "price": "$10", "inventory_quantity": 5>, <"title": "Screwdriver", "price": "$5", "inventory_quantity": 10>>
        output: "The hammer is available for $10 and we have 5 in stock."
        """,
    ),
    (
        "human",
        "{input}"
    ),
])



        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )

        response = chain.invoke({"input": input_text})
        return response['text']
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise


# Store chat history
chat_history = []

    
@app.route('/check-availability', methods=['POST'])
def check_availability():
    try:
        message = request.json
        message.update({
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history)
        })
        chat_history.append(message)
        

        search_keywords = get_keywords(message['content'])
        print(keywords)

        squery = f"user query : {message['content']} and extracted keywords from user query:{search_keywords}"

        # related_products_for_query,forai = get_product_search(squery) 
        related_products_for_query,forai = get_product_search(search_keywords)

        query = f"user query : {message['content']} and related products based on user query:{str(forai)}"

        ai_response = get_availability(input_text = query)
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history),
            'related_products_for_query':related_products_for_query
        }

        chat_history.append(response)
        
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            'error_response': str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            # 'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    
@app.route('/chat-product-search', methods=['POST'])
def chat_product_search():
    try:
        message = request.json
        message.update({
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history)
        })
        chat_history.append(message)


        search_keywords = get_keywords(message['content'])
        print(keywords)
        squery = f"user query : {message['content']} and extracted keywords from user query:{keywords}"
        print(squery)

        # related_products_for_query,forai = get_product_search(squery)
        related_products_for_query,forai = get_product_search(search_keywords)

        ai_response = get_response_product_search(input_text = message['content'], related_products = str(forai))
        
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history),
            'related_products_for_query':related_products_for_query

        }
        chat_history.append(response)
        
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            "error_response" : str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        chat_history.clear()
        memory.clear()
        return jsonify({"message": "Chat history cleared successfully"})
    except Exception as e:
        return jsonify({"error": "Failed to clear chat history"}), 500
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "working"})

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))  # Render assigns a dynamic port
#     app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    app.run(debug=True)
