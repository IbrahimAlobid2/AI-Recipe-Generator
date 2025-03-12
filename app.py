import os
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from groq import Groq
import base64
import streamlit as st


load_dotenv(find_dotenv())
GROQ_API = os.getenv("GROQ_API_KEY")

model_name = 'llama-3.3-70b-versatile'
llm  =  ChatGroq(api_key=GROQ_API,
    model_name=model_name,
    temperature=0.0,
    
)

client = Groq(api_key= GROQ_API)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



def image_to_text(url):
    base64_image = encode_image(url)
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )
    text = completion.choices[0].message
    print(f"Image Captioning:: {text}")
    return text





def generate_recipe(ingredients):
    template = """
    You are a extremely knowledgeable nutritionist, bodybuilder and chef who also knows
                everything one needs to know about the best quick, healthy recipes. 
                You know all there is to know about healthy foods, healthy recipes that keep 
                people lean and help them build muscles, and lose stubborn fat.
                
                You've also trained many top performers athletes in body building, and in extremely 
                amazing physique. 
                
                You understand how to help people who don't have much time and or 
                ingredients to make meals fast depending on what they can find in the kitchen. 
                Your job is to assist users with questions related to finding the best recipes and 
                cooking instructions depending on the following variables:
                0/ {ingredients}
                
                When finding the best recipes and instructions to cook,
                you'll answer with confidence and to the point.
                Keep in mind the time constraint of 5-10 minutes when coming up
                with recipes and instructions as well as the recipe.
                
                If the {ingredients} are less than 3, feel free to add a few more
                as long as they will compliment the healthy meal.
                
            
                Make sure to format your answer as follows:
                - The name of the meal as bold title (new line)
                - Best for recipe category (bold)
                    
                - Preparation Time (header)
                    
                - Difficulty (bold):
                    Easy
                - Ingredients (bold)
                    List all ingredients 
                - Kitchen tools needed (bold)
                    List kitchen tools needed
                - Instructions (bold)
                    List all instructions to put the meal together
                - Macros (bold): 
                    Total calories
                    List each ingredient calories
                    List all macros 
                    
                    Please make sure to be brief and to the point.  
                    Make the instructions easy to follow and step-by-step .
    """
    prompt = PromptTemplate(template=template, input_variables=["ingredients"])
    recipe_chain = prompt | llm
    recipe = recipe_chain.invoke(ingredients)


    return recipe





def main():


    st.title("Image To Recipe 👨🏾‍🍳")
    st.header("Upload an image and get a recipe")

    upload_file = st.file_uploader("Choose an image:", type=["jpg", "png"])

    if upload_file is not None:
        print(upload_file)
        file_bytes = upload_file.getvalue()
        with open(upload_file.name, "wb") as file:
            file.write(file_bytes)

        st.image(
            upload_file,
            caption="The uploaded image",
            use_container_width=True,
            width=250
        )
        ingredients = image_to_text(upload_file.name)


        recipe = generate_recipe(ingredients=ingredients)

        with st.expander("Ingredients"):
            st.write(ingredients.content)
        with st.expander("Recipe"):
            st.write(recipe.content)




# Invoking main function
if __name__ == "__main__":
    main()
