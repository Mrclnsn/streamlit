import streamlit as st 
from langchain.chains import LLMChain, SimpleSequentialChain 
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate 

st.title("✅ What's TRUE  : Using LangChain `SimpleSequentialChain`")

st.markdown("Inspired from [fact-checker](https://github.com/jagilley/fact-checker) by Jagiley")


if API:
    llm = OpenAI(temperature=0.7, openai_api_key=API)
else:
    
    st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")


user_question = st.text_input(
    "Enter Your Question : ",
placeholder = "Cyanobacteria can perform photosynthetsis , are they considered as plants?",
)

if st.button("Tell me about it", type="primary"):
    
    template = """{question}\n\n"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)
    
template = """Here is a statement:
{statement}
Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
prompt_template = PromptTemplate(input_variables=["statement"], template=template)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
assumptions_chain_seq = SimpleSequentialChain(
chains=[question_chain, assumptions_chain], verbose=True
)

    
template = """Here is a bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
fact_checker_chain_seq = SimpleSequentialChain(
chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True
)

    
template = """In light of the above facts, how would you answer the question '{}'""".format(
user_question
)
template = """{facts}\n""" + template
prompt_template = PromptTemplate(input_variables=["facts"], template=template)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)
overall_chain = SimpleSequentialChain(
chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
verbose=True,
)

    
st.success(overall_chain.run(user_question))
