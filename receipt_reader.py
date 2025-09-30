def main():
    import os
    import base64
    import pandas as pd
    from tqdm import tqdm
    from pydantic import BaseModel, Field
    from langchain_ollama import ChatOllama
    
    # Define the structured output model
    class Receipt(BaseModel):
        shop_name: str = Field(description="The name of the shop")
        transaction_date: str = Field(description="The date of the transaction, output in format %d-%m-%Y")
        net_amt: float = Field(description="The net amount of the sale transaction")
    
    # Initialize DataFrame
    df = pd.DataFrame(columns=['File','Shop name', 'Transaction date', 'Net Amount'])
    
    # Initialize the language model
    llm = ChatOllama(model="mistral-small3.2:24b", temperature=0)
    structured_llm = llm.with_structured_output(Receipt)
    
    # Directory containing receipt images
    directory = "images/"
    filenames = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    counter = 0
    
    # Process each image
    for filename in tqdm(filenames):
        try:
            with open(directory+filename, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
                message = {
                    "role":"user",
                    "content": [
                        {
                            "type":"text",
                            "text":"""You are a Optical Character Recognition operator. You will extract the following information from the receipt images.
                            - shop name
                            - transaction date (typically shown as DD/MM/YY or DD/MM/YYYY)
                            - total net amount payable."""
                        },
                        {
                            "type":"image",
                            "source_type":"base64",
                            "data": image_data,
                            "mime_type":"image/jpeg"
                        }
                    ]
                }
                response = structured_llm.invoke([message])
                df.loc[counter] = [filename,response.shop_name,response.transaction_date,response.net_amt]
                counter+=1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Convert date column and export to CSV
    df['Transaction date'] = pd.to_datetime(df['Transaction date'])
    df.to_csv('output.csv', index=False)

if __name__ == "__main__":
    main()
