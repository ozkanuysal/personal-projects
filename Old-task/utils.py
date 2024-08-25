import textwrap
import google.generativeai as genai
from typing import List
from app.schemas.adcopy import AdCopyParameters 
from app.schemas.audience import AudienceParameters
from sqlalchemy.orm import Session
from fastapi import Depends,HTTPException
from app.models import Adcopy
import json

def get_model():
    """Returns a generative model."""
    return genai.GenerativeModel('gemini-pro')

def generate_content(prompt, parameters):
    """Generates content using Gemini with prompt and parameters."""
    model = get_model()
    prompt = prompt.format(**parameters)  
    response = model.generate_content(prompt)
    content_parts = response.candidates[0].content.parts
    text = ''.join(part.text for part in content_parts)
    return {"Text": text}

def generate_ad_contents(db: Session, adcopy_parameters: List[AdCopyParameters]):
    """Generates ad content with product name, price, and other parameters."""
    contents =  [
        {
            "Generated Content": generate_content(
                f"Create an ad copy for a product named '{product.product_name}', which is priced at ${product.product_price}. The ad copy should highlight the unique features and benefits of the product, as described here: '{product.product_description}'. It should also mention that the product belongs to the '{product.product_category}' category. For style and tone, refer to this example: '{product.ad_copy_example}'. Additionally, incorporate the following information into the ad copy: '{product.additional_info}'. The ad copy should be a single sentence that is both creative and informative. This ad copy will be used for a Facebook ad campaign.",
                product.dict()
            )
        }
        for product in adcopy_parameters
    ]
    contents_str = json.dumps(contents)
    adcopy_content = Adcopy(content=contents_str)
    db.add(adcopy_content)
    db.commit()

    return contents



def generate_audience_contents(audience_parameters: List[AudienceParameters]):
    """ Generates audience content with various parameters."""
    return [
        {
            "Generated Content": generate_content(
                f"Create an ad copy targeting an audience with the following characteristics: They are interested in {param.audience_Interests}, their demographic profile is {param.audience_Demographics}, and they do not meet the following exclusion criteria: {param.audience_exclusion_criteria}. The ad copy should be concise, not exceeding one sentence. It should utilize data from {param.audience_data_source} and should be similar in style and tone to this example: {param.audience_example}.",
                param.dict()
            )
        }
        for param in audience_parameters
    ]
def get_ad_content_for_audience(db: Session, ad_content_id: int):
    """Retrieves ad content by ID and generates audience content."""
    adcopy_content = db.query(Adcopy).get(ad_content_id)
    if adcopy_content is None:
        raise HTTPException(status_code=404, detail="Ad content not found")
    ad_content = json.loads(adcopy_content.content)

    return {"Ad Content": ad_content}