from LLM import *
from pydanticClasses import *
import pandas as pd
from dataOperations import *
import json

instr = '''
You are an expert at crafting news articles that are challenging for advanced fake news detection models (e.g., MDFEND, RoBERTa) to classify. 
Your task is to subtly modify the content and comments of news articles in JSON format to retain their misleading intent while evading detection. 

Use the following strategies:
- Rephrase content to obscure clear domain-specific features, making it difficult to associate the article with a specific domain (e.g., blend political and entertainment language).
- Introduce hypothetical or opinion-based framing (e.g., 'It is reported...' or 'Experts suggest...') while maintaining plausibility.
- Mimic credible sources by using professional, journalistic tones and neutral or balanced language.
- Add plausible but unverifiable details to increase complexity and reduce specific cues (e.g., avoid explicit dates, locations, or statistics).
- If there are any names of people mentioned in the news, keep them as they are.
- Disrupt sentence patterns by introducing diverse structures, conflicting tones, or unrelated details to dilute semantic coherence.
- Modify comments to include conflicting interpretations or perspectives (e.g., supportive, critical, and neutral), creating a mix of sentiments.
- Where applicable, create cross-domain references or introduce ambiguity in the metadata (e.g., timestamps or references to unrelated events).
- Ensure all fields (content, comments, and category) are populated. The "category" field must remain unchanged.

Do not give any intro or outro, or explanation etc. All the three are strings like regular text, and they should not contain weird start or stop characters.
'''

def processBatch(llm, batch_json, instr, temperature, responseModel):
    prompt = f"""
    Modify the content and comments but not category in each of these articles I provide below. Do not truncate in this process.
    Articles (JSON):
    {batch_json}
    """
    try:
        response = llm.llmResponse(temperature, instr, prompt, responseModel)
        if hasattr(response, "articles") and response.articles:
            return [article.model_dump() for article in response.articles]
        else:
            return []
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []


def modifyWithGemini(df, batch_size=3, temperature=0.35, max_retries=2):
    llm = GoogleLLM()
    llm.reAuth()

    OUTPUT_FILE = f"modifiedData/{llm.name}_temp_{temperature}.json"
    
    modified_data = []
    failed_batches = []

    for batch_index, start_idx in enumerate(range(0, len(df), batch_size)):
        batch_df = df.iloc[start_idx : start_idx + batch_size]
        batch_json = batch_df.to_json(orient="records")
        print(f"Processing batch {batch_index + 1}...")

        if batch_index % 6 == 0:
            llm.reAuth()

        retries = 0
        processed_articles = None

        while retries < max_retries:
            try:
                processed_articles = processBatch(
                    llm=llm,
                    batch_json=batch_json,
                    instr=instr,
                    temperature=temperature,
                    responseModel=ArticleArray
                )

                if processed_articles:
                    modified_data.extend(processed_articles)
                    break
            except Exception as e:
                print(f"Error processing batch {batch_index + 1}, attempt {retries + 1}: {e}")

            retries += 1
            if retries < max_retries:
                print(f"Retrying batch {batch_index + 1} (attempt {retries + 1}/{max_retries})...")
        
        if retries == max_retries and not processed_articles:
            print(f"Batch {batch_index + 1} failed after {max_retries} attempts.")
            failed_batches.append(batch_index)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(modified_data, f, indent=4)
    print(f"Modified data saved to {OUTPUT_FILE}.")

    if failed_batches:
        print(f"Failed batches: {failed_batches}")
    else:
        print("All batches processed successfully.")

if __name__ == "__main__":
    dataOps = DataOperations()
    df = dataOps.preProcess()
    modifyWithGemini(df, batch_size=3, temperature=0.35)