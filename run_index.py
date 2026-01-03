from src.vectorstore import vectorstore
from src.config import load_dotenv
import traceback

load_dotenv()

print("Starting Pinecone Indexing Process")
print("Checking for index and loading data if necessary...")

try:
    vs = vectorstore()
    print("Pinecone Indexing successful.")
    print("Data telah berhasil dimasukkan ke index Pinecone.")

except Exception as e:
    print("\n Error during indexing!")
    print("Detail error:")
    print(traceback.format_exc())
    print("\n Pastikan:")
    print("1. File PDF berada di: data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
    print("2. PINECONE_API_KEY benar dan aktif di lingkungan .env")
    print("3. Folder data/ benar-benar berisi file PDF")
