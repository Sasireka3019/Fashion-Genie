import streamlit as st
import random
import time
import spacy
import base64
from PIL import Image
import os 
import cv2
import numpy as np
import pandas as pd
import pickle
import tempfile
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from user_profiling import get_user_preferences
from trend_extractor import PinterestImageScraper  # Import the PinterestImageScraper class
from genai_func import build_metadata_encoder, build_image_encoder, image_encoder
p_scraper_instance = PinterestImageScraper()

st.set_page_config(page_title='Fashion Genie')
ner_model = spacy.load("spacy_model_chatbot")
latent_dim = 100
image = Image.open("background.jpg")
new_image = image.resize((700, 200))
st.image(new_image)

title = '<p style="font-family:Georgia;color:purple;font-size:50px;text-align:center;">Fashion Genie</p>'
st.markdown(title, unsafe_allow_html=True)

def generate_noise():
    image_size = (64, 64)
    noise_image = np.random.randint(0, 256, size=(image_size[0], image_size[1], 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.close()  # Close the temporary file to allow image loading
        img_to_save = Image.fromarray(noise_image)
        img_to_save.save(temp_filename)
    return temp_filename

def get_first_image_from_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add more extensions if needed
    image_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)) and any(file.lower().endswith(ext) for ext in image_extensions)]
    
    if image_files:
        return os.path.join(folder_path, image_files[0])
    else:
        return None

def get_attributes(entities):
    need = {}
    extracted_attributes = {
        "gender": None,
        "age": None,
        "weather": "summer",
        "region": None,
        "top type": "tops",
        "bottom type": "jeans",
        "bag type": "purse",
        "shoe type": "heals",
        "handwear": "bangle",
        "occasion": "party",
        "color1": None,
        "color2": None,
        "style": None,
        "material": None,
        "sleeve": "yes"
    }
    user_id = 1
    preferences = get_user_preferences(user_id)
    extracted_attributes["color1"] = preferences["color"]
    extracted_attributes["style"] = preferences["style"]
    extracted_attributes["region"] = preferences["location"]
    extracted_attributes["gender"] = preferences["gender"]
    extracted_attributes["material"] = preferences["material"]

    flag = 0
    if(extracted_attributes["color1"]):
        flag = 1
    
    for entity in entities:
        entity_type = entity[1]
        entity_value = entity[0]
        entity_type = entity_type.replace("-", " ")
        entity_type = entity_type.replace("outfit type", "top type")
        if(flag == 0):
            entity_type = entity_type.replace("color", "color1")
            flag = 1
        else:
            extracted_attributes["color2"] = extracted_attributes["color1"]
            entity_type = entity_type.replace("color", "color1")
        if(entity_type == "occasion"):
            entity_value = entity_value.replace("marriage", "party")
            entity_value = entity_value.replace("wedding", "party")
            entity_value = entity_value.replace("business", "formal")
            entity_value = entity_value.replace("meeting", "formal")
            entity_value = entity_value.replace("vacation", "casual")
            entity_value = entity_value.replace("festival", "party")
            entity_value = entity_value.replace("festive", "party")
            entity_value = entity_value.replace("cultural", "party")
            entity_value = entity_value.replace("traditional", "party")
            entity_value = entity_value.replace("night out", "party")
            entity_value = entity_value.replace("picnic", "casual")
            entity_value = entity_value.replace("outing", "casual")
            entity_value = entity_value.replace("event", "formal")
            entity_value = entity_value.replace("everyday", "casual")
            entity_value = entity_value.replace("special", "party")
            entity_value = entity_value.replace("office", "formal")
            entity_value = entity_value.replace("Diwali", "party")
            entity_value = entity_value.replace("trip", "casual")
            entity_value = entity_value.replace("vibe", "casual")
            if entity_value not in ["formal", "casual", "party"]:
                entity_value = "party"
        if(entity_type == "sleeve"):
            if "full" in entity_value:
                entity_value = "full"
            elif "less" in entity_value:
                entity_value = "no"
            else:
                entity_value = "yes"
        if(entity_type == "weather"):
            if entity_value in ["summer", "winter", "rainy"]:
                entity_value = entity_value
            elif entity_value in ["cold", "chill"]:
                entity_value = "winter"
            elif entity_value in ["hot", "heat", "warm"]:
                entity_value = "summer"
            else:
                entity_value = "rainy"
        try:
            extracted_attributes[entity_type] = entity_value
            need[entity_type] = entity_value
        except:
            pass
    print(extracted_attributes)
    return [extracted_attributes, need]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

welcome_message = "Welcome to Fashion Genie! Your personal fashion assistant. How can I help you today?"
with st.chat_message("assistant"):
    st.markdown(welcome_message)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process NER on user input
    doc = ner_model(prompt)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    user_preferences = {}
    metadata_encoder = None
    image_embedding = None
    if entities:
        data = get_attributes(entities)
        user_preferences = data[0]
        need = data[1]
        # Load the label_encoders dictionary from the file
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        lst = ["female", 20, "summer", "america", "chudi", "jeans", "purse", "heals", "watch", "party", "blue", "white", "floral", "cotton", "yes"]
        cols = ["gender", "age", "weather", "region", "top type", "bottom type", "bag type", "shoe type", "handwear", "occasion", "color1", "color2", "style", "material", "sleeve"]
        i = 0
        input_metadata = pd.DataFrame()
        for column in user_preferences:
            if column in label_encoders:
                le = label_encoders[column]
                try:
                    encoded_value = le.transform([user_preferences[column]])
                except:
                    encoded_value = -1
                input_metadata[column] = encoded_value
            i += 1
        num_metadata_features = len(input_metadata.columns) - 1
        metadata_encoder = build_metadata_encoder(num_metadata_features)
        entity_response = f"Please wait for an amazing outfit with...{need}"
        # Display detected entities in chat message container
        with st.chat_message("assistant"):
            st.markdown(entity_response)
    
    if user_preferences:
        tops = "trending " + user_preferences["region"] + " " + user_preferences["color1"] + " " + user_preferences["top type"]
        bottoms = "trending " + user_preferences["region"] + " " + user_preferences["color1"] + " " + user_preferences["bottom type"]
        shoe = "trending " + user_preferences["region"] + " " + user_preferences["color1"] + " " + user_preferences["shoe type"]
        bag = "trending " + user_preferences["region"] + " " + user_preferences["color1"] + " " + user_preferences["bag type"]
        handwear = "trending " + user_preferences["region"] + " " + user_preferences["color1"] + " " + user_preferences["handwear"]
        print(tops, bottoms, shoe, bag, handwear)
        download_tops = p_scraper_instance.make_ready(tops)
        download_bottoms = p_scraper_instance.make_ready(bottoms)
        download_shoe = p_scraper_instance.make_ready(shoe)
        download_bag = p_scraper_instance.make_ready(bag)
        download_handwear = p_scraper_instance.make_ready(handwear)
        st.markdown("### Here are some trending outfits")
        if download_tops:
            try:
                tops_path = "_".join(tops.split(" "))
                tops_image = get_first_image_from_folder(tops_path)
                # st.image(tops_image)
                image = Image.open(tops_image)
                new_image = image.resize((200, 200))
                st.image(new_image)
            except:
                tops_image = generate_noise()
        else:
            tops_image = generate_noise()
        
        if download_bottoms:
            try:
                bottom_path = "_".join(bottoms.split(" "))
                bottoms_image = get_first_image_from_folder(bottom_path)
                image = Image.open(bottoms_image)
                new_image = image.resize((200, 200))
                st.image(new_image)
            except:
                bottoms_image = generate_noise()
        else:
            bottoms_image = generate_noise()
        # st.image(bottoms_image)
        
        if download_shoe:
            try:
                shoe_path = "_".join(shoe.split(" "))
                shoe_image = get_first_image_from_folder(shoe_path)
                # st.image(shoe_image)
                image = Image.open(shoe_image)
                new_image = image.resize((200, 200))
                st.image(new_image)
            except:
                shoe_image = generate_noise()
        else:
            shoe_image = generate_noise()
        
        if download_bag:
            try:
                bag_path = "_".join(bag.split(" "))
                bag_image = get_first_image_from_folder(bag_path)
                # st.image(bag_image)
                image = Image.open(bag_image)
                new_image = image.resize((200, 200))
                st.image(new_image)
            except:
                bag_image = generate_noise()
        else:
            bag_image = generate_noise()
        
        if download_handwear:
            handwear_path = "_".join(handwear.split(" "))
            handwear_image = get_first_image_from_folder(handwear_path)
            image = Image.open(handwear_image)
            new_image = image.resize((200, 200))
            st.image(new_image)
        else:
            handwear_image = generate_noise()
        image_paths = [tops_image, bottoms_image, shoe_image, bag_image, handwear_image]
        image_size = (64, 64)
        outfit_images = [img_to_array(load_img(img_path, target_size=image_size)) for img_path in image_paths]
        normalized_outfit_images = [img / 255.0 for img in outfit_images]
        normalized_outfit_images = np.array(normalized_outfit_images)
        normalized_outfit_images = np.expand_dims(normalized_outfit_images, axis=0)
        num_samples = 1 
        noise = np.random.normal(0, 1, (num_samples, latent_dim))
        sampled_metadata = input_metadata
        sampled_metadata = sampled_metadata.astype(np.float32)
        image_embeddings = image_encoder.predict(normalized_outfit_images)  # Encode image metadata
        generator = load_model("generator_model.h5")
        generated_outfits = generator([noise, image_embeddings, sampled_metadata], training=False)
        generated_outfits = (generated_outfits + 1) / 2.0

        st.markdown("### Generated Outfit")
        plt.figure(figsize=(15, num_samples * 2))  # Adjust figsize based on the number of samples

        for j in range(num_samples):
            for k in range(generated_outfits.shape[1]):
                plt.subplot(num_samples, generated_outfits.shape[1], j * generated_outfits.shape[1] + k + 1)
                plt.imshow(generated_outfits[j, k])
                plt.axis('off')
                plt.title(f"Generated Item {k+1}")

        plt.tight_layout()
        plt.show()


# # Convert the generated outfits EagerTensor to a NumPy array
#         generated_outfits_np = generated_outfits.numpy()

#         # Loop through the generated outfits
#         for j in range(num_samples):
#             st.write(f"Generated Outfit {j+1}:")
            
#             # Loop through the different items in the outfit (e.g., tops, bottoms, etc.)
#             for k in range(generated_outfits_np.shape[1]):
#                 # Get the image data for the current item
#                 outfit_image_data = generated_outfits_np[j, k]
                
#                 # Convert image data to a PIL Image
#                 outfit_pil_image = Image.fromarray((outfit_image_data * 255).astype('uint8'))
                
#                 outfit_pil_image = Image.fromarray((outfit_image_data * 255).astype('uint8'))
        
#                 # Resize the PIL Image to the desired dimensions
#                 resized_outfit_pil_image = outfit_pil_image.resize((200, 200))
                
#                 # Display the resized PIL Image using Streamlit
#                 st.image(resized_outfit_pil_image, caption=f"Generated Item {k+1}", use_column_width=True)
#                 # Display the PIL Image using Streamlit
#                 # st.image(outfit_pil_image, caption=f"Generated Item {k+1}", use_column_width=True)

#         # Display generated outfit images
#         # st.image(generated_outfits, width=100, use_column_width=False, caption=[f"Generated Item {k+1}" for k in range(generated_outfits.shape[1])], channels="RGB")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, human! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
