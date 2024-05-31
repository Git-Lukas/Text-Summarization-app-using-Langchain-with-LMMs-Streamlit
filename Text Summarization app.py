import streamlit as st
from langchain.text_splitter import CharacterTextSplitter # used for splitter the text into smalle chunks
from langchain.docstore.document import Document # convert the chunks in document format
from langchain.chains.summarize import load_summarize_chain # connect prompt and llm model
from langchain import PromptTemplate # for creating prompt 
from langchain.llms import CTransformers # loading the llm model
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Function to summarize an input text
def summarize_text(txt):
    # We instantiate the callback with a streaming stdout handler
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])   

    # loading the LLM model
    # This open source model can be downloaded from here
    # Their are multiple models available just replace it in place of model and try it.
    llm = CTransformers(
        model=r"llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
         max_new_tokens = 512,
        temperature = 0.5   )

    # text splitter method by default it has chunk_size = 200 and chunk_overlap = 200
    text_splitter = CharacterTextSplitter() 
    # split the text into smaller chunks
    texts = text_splitter.split_text(txt) 
    # convert the splitted chunks into document format
    docs = [Document(page_content=t) for t in texts] 
    # Text summarization
    chain = load_summarize_chain(llm,chain_type='map_reduce')
    return chain.run(docs)


# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            response = summarize_text(txt_input)
            result.append(response)


if len(result):
    st.title('📝✅ Summarization Result')
    st.info(response)
