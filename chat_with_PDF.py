import os
import streamlit as st
from dotenv import load_dotenv

# Libraries for Reading PDF and Text Splitting.
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Libraries from Lang Chain
import google.generativeai as genai     # This header helps to create generative AI model from Google Gemini-AI.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS    # This package is for vector embeddings.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain    # This package does the chats or any kind of prompts.
from langchain.prompts import PromptTemplate        # This helps you to get prompt template.


class MainClass(object):

    @staticmethod
    def perform_configurations():

        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

    @staticmethod
    def get_pdf_text(pdf_docs):

        text = ""

        for pdf in pdf_docs:

            # We are reading the PDF here.
            pdf_reader = PdfReader(pdf)

            # As soon as we read the PDF, it should get the detail of all the pages.
            # So, this pdf_reader.pages variable will be in list format. Each page is each item in the list.
            for page in pdf_reader.pages:

                # It means, we are going the extract all the text inside that page.
                text += page.extract_text()

        return text

    @staticmethod
    def get_text_chunks(text):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)

        return chunks

    @staticmethod
    def get_vector_store(text_chunks):

        # Here, we are using Google Generative AI embeddings.
        # There are lot of embeddings are there, like Hugging Face and Open AI provides different one.
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        # Vector store can be stored in a local environment or in a database.
        # Now, we are storing it in local
        # The name of the local folder will be "faiss_index"
        # In this location, we can see our Vectors and it is not readable.
        vector_store.save_local("faiss_index")

    @staticmethod
    def get_conversational_chain():

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if
        the answer is not in provided context just say, "answer is not available in the context", don't provide the 
        wrong answer\n\n
        Context: \n {context}?\n
        Question: \n {question}?\n
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain

    @staticmethod
    def user_input(user_question):

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # We will be loading faiss_index from local.
        # The PDFs we uploaded are in the form of Vector in the folder of faiss index.
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)

        chain = MainClass.get_conversational_chain()

        response = chain(
            {'input_documents': docs, 'question': user_question},
            return_only_outputs=True
        )

        print(response)

        st.write("Reply: ", response['output_text'])



def main():

    # Loading environment variables and setting up passwords in the environment
    MainClass.perform_configurations()

    # Creating Streamlit application (Front-end)
    st.set_page_config('Chat With Multiple PDF')
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input('Ask a question from the PDF files.')

    if user_question:
        MainClass.user_input(user_question)

    with st.sidebar:

        st.title('Menu:')
        pdf_docs = st.file_uploader("Upload your PDF files and Click on the ")

        if st.button('Submit & Process'):

            with st.spinner('Processing...'):

                # Step 1: Read from PDF
                # Reading the text from all the PDFs.
                raw_text = MainClass.get_pdf_text(pdf_docs)

                # Step 2: PDF to Chunks
                # We are making the text into smaller chunks which we read from PDF
                text_chunks = MainClass.get_text_chunks(raw_text)

                # Step 3: Chunks to Vector and Save it in Local.
                # After getting chunks, we can convert the chunks into vector and save it in local.
                MainClass.get_vector_store(text_chunks)

                st.success('Done')


if __name__ == '__main__':
    main()
