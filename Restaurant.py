import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# ---------------- UI ----------------
st.set_page_config(page_title="Restaurant Name Generator", layout="centered")
st.title("🍽️ Restaurant Name Generator")

cuisine = st.sidebar.selectbox(
    "Pick a Cuisine",
    ("Indian", "Italian", "Mexican", "Arabic", "American"),
    key="cuisine_selectbox"
)

generate = st.sidebar.button(
    "Generate",
    key="generate_btn"
)


# ---------------- LLM ----------------
llm = ChatOpenAI(
    temperature=0.7
)

# ---------------- Prompts ----------------
prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Tell me a fancy restaurant name."
)

prompt_template_items = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Tell some menu items for the restaurant {restaurant_name}. Return ONLY the items with numbers like 1.,2. etc.., each on a new line."
)

# ---------------- Sequential Chain ----------------
def sequential_chain(inputs):
    cuisine = inputs["cuisine"]

    restaurant_name = llm.invoke(
        prompt_template_name.format(cuisine=cuisine)
    ).content.strip()

    menu_items = llm.invoke(
        prompt_template_items.format(restaurant_name=restaurant_name)
    ).content.strip()

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

chain = RunnableLambda(sequential_chain)

# ---------------- Run App ----------------
if generate:
    with st.spinner("Generating restaurant details..."):
        response = chain.invoke({"cuisine": cuisine})

    st.success("Done!")

    st.header(response["restaurant_name"])
    st.subheader("📜 Menu Items")

    menu_items = response["menu_items"].split(",")
    for item in menu_items:
        st.write(item.strip())