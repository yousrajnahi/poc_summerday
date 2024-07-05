import nltk
import os
import tempfile
import shutil
import pandas as pd
import subprocess

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from langchain_community.document_loaders import PDFMinerLoader, DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from pdfminer.high_level import extract_text
from langchain_community.document_loaders import (UnstructuredFileLoader,
                                                  PDFMinerLoader,
                                                  CSVLoader,
                                                  JSONLoader,
                                                  TextLoader,
                                                  UnstructuredXMLLoader,
                                                  UnstructuredHTMLLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredEmailLoader)


def organize_files_by_extension(source_directory):
    # Initialize an empty dictionary to hold files organized by their extensions
    organized_dict = {}
    # Walk through all the directories and files in the source directory
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            # Get the full file path
            file_path = os.path.join(root, filename)
            # Split the file name to get the extension
            _, extension = os.path.splitext(filename)
            # Check if the file has an extension and is not a hidden file
            if extension and not filename.startswith('.'):
                # Normalize the extension by removing the leading dot and converting to lowercase
                extension_folder = extension[1:].lower()
                # If the extension is not already a key in the dictionary, add it with an empty list
                if extension_folder not in organized_dict:
                    organized_dict[extension_folder] = []
                # Append the file path to the list corresponding to its extension
                organized_dict[extension_folder].append(file_path)
    # Return the dictionary containing files organized by their extensions
    return organized_dict


def convert_files_in_directory(data_directory):
    # Browse all files in the directory
    for file in os.listdir(data_directory):
        # Full file path
        file_path = os.path.join(data_directory, file)
        # Check if the file is a regular file
        if os.path.isfile(file_path):
            # Get the file extension
            extension = file.split(".")[-1].lower()
            # Check if the extension is in the list for PDF conversion
            if extension in ["epub", "rtf", "doc", "odt", "ppt", "docx", "pptx"]:
                # Convert to PDF
                subprocess.run(['libreoffice', '--convert-to', 'pdf:writer_pdf_Export', file_path, '--outdir', data_directory])
                # Delete the old file
                os.remove(file_path)
            # Check if the extension is for Excel files
            elif extension in ["xlsx", "xls"]:
                # Convert to CSV
                pd.read_excel(file_path).to_csv(os.path.join(data_directory, f"{os.path.splitext(file)[0]}.csv"), index=False)
                # Delete the old file
                os.remove(file_path)

def load_documents(DATA_PATH, glob_pattern,loader_class, loader_args):
  # Create a DirectoryLoader instance with specified parameters.
    loader = DirectoryLoader(DATA_PATH, glob=glob_pattern, show_progress=True,use_multithreading=True,loader_cls = loader_class, loader_kwargs =loader_args)
    # Load the documents using the configured loader and return them.
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def save_uploaded_files(uploaded_files, directory):
    for uploaded_file in uploaded_files:
        # Path for the new file in the specified directory
        temp_file = os.path.join(directory, uploaded_file.name)
        # Write the uploaded file to the new file on disk
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            
def create_directory(directory_path):
    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
def remove_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

def read_template_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
