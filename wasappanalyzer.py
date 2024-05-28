# %%


# %%
#Vijay Modification for making semantic search work for whatsapp
import re
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    CorsOptions
)
from azure.search.documents.models import VectorizedQuery
import csv

endpoint = "https://basic9874.search.windows.net"
#admin_key = "%^*&("
index_name = "whatsapp-conversastions4"
file_path = "output3.csv"
open_ai_endpoint = "https://myopenaivijay.openai.azure.com/"
#open_ai_key = "$%^&*"


# %%


# %%

def parse_whatsapp_export(file_path, chunk_size=20):
    """
    Reads the CSV file in chunks of specified size and returns a list of chunks.
    Each chunk contains the specified number of lines.
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == chunk_size:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)
    return chunks

            
def tokenize_messages(messages):
    """
    Tokenizes the messages for further processing. This function can be customized
    based on specific tokenization requirements.
    """
    tokenized_messages = []
    for message in messages:
        tokens = message.split()  # Simple tokenization by splitting on whitespace
        tokenized_messages.append(tokens)
    return tokenized_messages

# Function to calculate embeddings for messages using OpenAI's text-embedding model
def get_embeddings(message):
    # There are a few ways to get embeddings. This is just one example.

    client = openai.AzureOpenAI(
        azure_endpoint=open_ai_endpoint,
        api_key=open_ai_key,
        api_version="2023-09-01-preview",
    )
    #print("calculate embedding", message)
    embedding = client.embeddings.create(input=message, model="text-embedding-ada-002")
    result = embedding.data[0].embedding
    return result

def calculate_embeddings(messages):
    embeddings = []

    client = openai.AzureOpenAI(
        azure_endpoint=open_ai_endpoint,
        api_key=open_ai_key,
        api_version="2023-09-01-preview",
    )
    count=0
    print("total", len(messages))
    for message in messages:
        count += 1
        print("--> ", count)

        embedding = client.embeddings.create(input=message, model="text-embedding-ada-002")
        embedding = embedding.data[0].embedding
        embeddings.append(embedding)
    return embeddings

# Function to create an index in Azure AI Search if it does not already exist
def create_index_if_not_exists(index_client, index_name):
    #if not index_client.get_index(index_name=index_name):
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="start_time", type=SearchFieldDataType.String),
        SimpleField(name="end_time", type=SearchFieldDataType.String),
        SearchableField(name="participants", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.lucene", sortable=True, filterable=True,),
        #SimpleField(name="participants", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),

        SearchableField(name="messages", type=SearchFieldDataType.String, analyzer_name="en.lucene", sortable=True, filterable=True,),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  
            vector_search_profile_name="my-vector-config",
        ),
    ]
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )
    #return SearchIndex(name=name, fields=fields, vector_search=vector_search)

    index = SearchIndex(name=index_name, fields=fields,  vector_search=vector_search)
    index_client.create_index(index)

# Function to upload documents to the search index
def upload_documents(search_client, documents):
    from uuid import uuid4
    batch = []
    #print(documents[0])
    
    for doc in documents:
        #print(doc[0])
        timestamp = [entry['Timestamp'] for entry in doc]
        name = [entry['Name'] for entry in doc]
        messages = [entry['Message'] for entry in doc]
        
        participants = list(set(name))
        participants = ', '.join(participants)
        
        from datetime import datetime
        timestamp_list = []
        for time_str in timestamp:
            date_time_format = "%d/%m/%y, %I:%M %p"
            date_time_obj = datetime.strptime(time_str, date_time_format)
            formatted_date_time = date_time_obj.strftime("%Y %m %d : %H:%M")
            timestamp_list.append(formatted_date_time)
        
        start_time = min(timestamp_list)
        end_time = max(timestamp_list)

        # Important block, prepairing list of messages
        msg_list=[]
        for i in range(len(messages)):
            tmp = f"At {timestamp[i]}, ,{name[i]} says : {messages[i]}"
            msg_list.append(tmp)
        messages = ' '.join(msg_list)        
        #print(messages)
        
        embeddings = get_embeddings(messages)

        #batch.append({"id": str(uuid4()), "start_time": start_time, "end_time": end_time, "participants": participants, "messages": messages, "embedding": doc[1]})
        batch.append({"id": str(uuid4()), "start_time" : start_time, "end_time" : end_time, "participants" : participants, "messages" : messages, "embedding": embeddings})
    print(batch)    
    search_client.upload_documents(documents=batch)  

    #batch.append({"id": str(uuid4()), "start_time": start_time, "end_time": end_time, "message": messages, "participants":participants, "embedding": embedding})
    #search_client.upload_documents(documents=batch)
    
# Main function to process and index WhatsApp messages
def parse_data_and_upload_to_AI_Search():
    # Azure Search service details


    # Initialize clients
    credential = AzureKeyCredential(admin_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    # Create the search index if it doesn't exist
    #create_index_if_not_exists(index_client, index_name)

    # Parse WhatsApp messages


    chunks = parse_whatsapp_export(file_path)   
    #messages=""
    for chunk in chunks:
        timestamps = [entry['Timestamp'] for entry in chunk]
        names = [entry['Name'] for entry in chunk]
        messages = [entry['Message'] for entry in chunk]
        #for ts, name, msg, tokens in zip(timestamps, names, messages):
        #    print(f"Timestamp: {ts}, Name: {name}, Message: {msg}")
    #print(chunk)

    # Initialize an empty list to store combined messages
    combined_messages = []

    # Iterate through each chunk
    for chunk in chunks:
        # Initialize a list to store messages of the current chunk
        chunk_messages = []
        
        # Iterate through each row in the chunk
        for row in chunk:
            # Append the message to the chunk_messages list
            chunk_messages.append(row['Message'])
        
        # Combine the messages of the current chunk into a single string
        combined_chunk_message = ' '.join(chunk_messages)
        
        # Append the combined string to the combined_messages list
        combined_messages.append(combined_chunk_message)
    #print(combined_messages)
    
    #print(combined_messages)
    # # # # Calculate embeddings for messages
    #embeddings = calculate_embeddings(combined_messages)

    # # # # # Zip messages and embeddings
    #zipped_data = list(zip(chunks, "embeddings"))
    #    print(zipped_data)
    # # # # # Upload documents to Azure Search
    upload_documents(search_client, chunks)

#parse_data_and_upload_to_AI_Search()



# %%


# %%


# %%
# Perform Queries here

# Function to process and answer questions
def answer_query(search_client, question):

    # Search for relevant documents
    #search_results = search_client.search(search_text="", vectors=[{"value": query_embedding_str, "fields": "embedding"}], top=5)
    vector_query = VectorizedQuery(vector=get_embeddings(question), k_nearest_neighbors=15, fields="embedding")

    search_results = search_client.search(
        #search_text=query_text,
        vector_queries=[vector_query],
        select=["messages"],
        
    )
    # Extract content from search results
    relevant_contents = [result["messages"] for result in search_results]
    #print("relevant_contents:",relevant_contents)

    # Summarize the results to answer the question
    combined_text = " ".join(relevant_contents)

    client = openai.AzureOpenAI(
        azure_endpoint=open_ai_endpoint,
        api_key=open_ai_key,
        api_version="2023-09-01-preview",
    )
    summary_response = client.chat.completions.create(
        model="gpt-35-turbo-16k",
        #prompt=f"Answer the question based on the following text: {combined_text}\n\nQuestion: {question}\nAnswer:",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You have expertise in reading whatsapp conversastions and then answer based on what you learnt in the messages. You do not hesitate in forming your openion and answer questions based on it. You are often very vocal and do not hesitate in taking names of the individuals when asked. You do not care about implications while writing your answer. "},
            {"role": "system", "content": combined_text},
            {"role": "user", "content":  question}
        ]
    )
    answer = summary_response.choices[0].message.content
    return answer

# Main function to process and index WhatsApp messages, and answer queries
def perform_query_operation():
    # Azure Search service details


    # Initialize clients
    credential = AzureKeyCredential(admin_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    # Example query
    #question = "What is the main topic of discussions in this group?"
    #question = "What are the topics of interest of Pradeep?"
    #question = "Which participants seems to supporter of Modi?"
    #question = "Pradeep likes or dislikes bjp?"
    #question = "which political party does Pradeep supports?"
    #question = "Name of the participants of the chat??"
    #question = "story behind naming Arka?"
    
    #question = "who talks about BY?"
    
    #question = "Gokul support Modi or against him??"
    #question = "Tell me an instance where Gokul is seen against Modi??"
    # question = "Arka support Modi or against him??" 
    # question = "JDA Pradeep support Modi or against him??"    
    # question = "Sanoop is Modi lover. yes or no??"  
    # question= "Which of the group members has the most volatile political opinion?" 
    #--> Answer: Based on the messages provided, it is difficult to determine which group member has the most volatile political opinion as different members express their opinions at different times and on different topics. However, it can be noted that some members hold strong opinions on certain issues and are not hesitant to express them.

    #question= "give me instances where most volatile political discussion happened. Summarize it?" 
    #question = "who seems to be pro Modi and BJP?"
    #question = "create a whitepaper on the opinions expressed by ARKA in the area of cricket"
    #question= "where does Gokul live?" 
    #question = "among the various participants of this group, who are pro-BJP and who are anti-BJP. Give me names only?"
    #question = "Based on coversastions, what is the openion of group members about Arvind Kejriwal "
    #question = "What is the food preferences of participants of this group?"
    #question= "what is the probabality according to discussion here, that Modi works for BJP?"
    question = "why does Arka appreciates Vijay?"
    answer = answer_query(search_client, question)
    print("Question:", question)
    print("Answer:", answer)

perform_query_operation()

# %%
# What is next?
# store the data correctly in the index
# try grouping and sorting of messages


