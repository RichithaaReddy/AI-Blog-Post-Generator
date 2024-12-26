import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM

# Load the model from local file (replace with your actual path)
model_path = r"C:\Users\RICHITHA REDDY\Downloads\llama-2-7b-chat.ggmlv3.q8_0.bin" 

try:
    # Attempt to load with ctransformers (specifying model_type as 'llama')
    model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama") 
except ValueError:
    # If ctransformers fails, try loading with transformers (if installed)
    try:
        from transformers import AutoModelForCausalLM 
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except ImportError:
        st.error("Error: Both ctransformers and transformers libraries are required. Please install them using `pip install ctransformers transformers`.")
        model = None  # Set model to None to prevent further attempts

def generate_blog_post(model, topic, num_words, style):
    

    if model is None:
        return "Error: Model loading failed."

    llm = CTransformers(model=model_path, model_type="llama", config={'max_new_tokens': 256, 'temperature': 0.01}) 
    template = f"Write a blog post on {topic} in {style} style with approximately {num_words} words."
    prompt = PromptTemplate(input_variables=['topic', 'style', 'num_words'], template=template)
    prompt = prompt.format(topic=topic, style=style, num_words=num_words)

    try:
        generated_text = llm(prompt) 
        # Split the generated text into a list of lines
        lines = generated_text.split('\n') 
        # Display each line separately using st.write()
        for line in lines:
            st.write(line) 
        return generated_text 
    except Exception as e:
        return f"Error during generation: {str(e)}"

def main():
    st.title("Blog Post Generator")

    # Get user input
    topic = st.text_input("Enter the topic of your blog:")
    num_words = st.number_input("Enter the desired number of words:", min_value=50, step=50)
    style = st.selectbox("Choose writing style:", ["Professional", "Humanized", "Simple"])


    if st.button("Generate"):
        generated_text = generate_blog_post(model, topic, num_words, style)
        #st.text_area("Generated Text", value=generated_text, height=200)

if __name__ == "__main__":
    main()

# import streamlit as st

# # Page configuration
# st.set_page_config(page_title="AI Chatbot", layout="wide")
# st.title("📄 AI Chatbot: PDF Knowledge Base + Chat System")

# # Sidebar for instructions
# with st.sidebar:
#     st.title("Instructions")
#     st.write("""
#     1. Upload a PDF file to use as the knowledge base.
#     2. Enter your questions in the chat system below to get replies based on the document.
#     """)

# # Session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # File uploader for PDF
# uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# if uploaded_file:
#     st.success("PDF uploaded successfully! Connect to the backend to process the document.")

# # Chat system
# st.header("💬 Chat with the AI Chatbot")

# user_question = st.text_input("Enter your question:", placeholder="Type your question here...")

# if user_question:
#     # Placeholder for backend connection
#     st.info("Fetching answer from the backend...")

#     # Simulated response (replace with your backend logic)
#     backend_response = "This is a simulated response. Connect this to your backend for real answers."

#     # Add to chat history
#     st.session_state.chat_history.append({"question": user_question, "answer": backend_response})

#     # Display chat history
#     st.subheader("Chat History")
#     for chat in st.session_state.chat_history:
#         st.write(f"**You:** {chat['question']}")
#         st.write(f"**Bot:** {chat['answer']}")
