import boto3
from botocore.exceptions import ClientError 
import re

PROMPT_FILE = "prompt-base.txt"

# stateful context, should be managed better
context = []

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

bedrock_runtime_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# load prompt from PROMPT_FILE
with open(PROMPT_FILE, "r") as f:
    baseline_prompt = f.read()

def build_context_prompt(tags) -> str:

    context_prompt = "<Live-playing-analysis>\n"

    if len(context) > 0:
        for block in context:
            context_prompt += "\t<player>\n"

            if len(block["audio_analysis_tags"]) > 0:
                context_prompt += "\t\t" + ",".join(block["audio_analysis_tags"]) + "\n"
            else:
                context_prompt += "n/a"

            context_prompt += "\t</player>\n"
            context_prompt += "\t<ia-drummer>\n"

            if block["proposed_prompt"] is not None:
                context_prompt += "\t\t" + block["proposed_prompt"] + "\n"

            context_prompt += "\t</ia-drummer>\n"
    
    context_prompt += "\t<player>\n"
    if len(tags) > 0:
        context_prompt += "\t\t" + ",".join(tags) + "\n"

    context_prompt += "\t</player>\n"
    context_prompt += "\t<ia-drummer>\n"

    return context_prompt

def invoke_claude(prompt, system_prompt):

    print("System prompt")
    print(system_prompt)
    print("User prompt")
    print(prompt)

    # Define the inference configuration
    inference_config = {
        "temperature": 0.5,  # Set the temperature for generating diverse responses
        "maxTokens": 200  # Set the maximum number of tokens to generate
    }
    # Define additional model fields
    additional_model_fields = {
        "top_p": 1,  # Set the top_p value for nucleus sampling
    }
    # Create the converse method parameters
    converse_api_params = {
        "modelId": model_id,  # Specify the model ID to use
        "messages": [{"role": "user", "content": [{"text": prompt}]}],  # Provide the user's prompt
        "inferenceConfig": inference_config,  # Pass the inference configuration
        "additionalModelRequestFields": additional_model_fields  # Pass additional model fields
    }
    # Check if system_text is provided
    if system_prompt:
        # If system_text is provided, add the system parameter to the converse_params dictionary
        converse_api_params["system"] = [{"text": system_prompt}]

    # Send a request to the Bedrock client to generate a response
    try:
        response = bedrock_runtime_client.converse(**converse_api_params)

        # Extract the generated text content from the response
        text_content = response['output']['message']['content'][0]['text']

        # Return the generated text content
        return text_content

    except ClientError as err:
        message = err.response['Error']['Message']
        print(f"A client error occured: {message}")

def get_prompt_from_tags(tags:list, tempo: int, bars: int) -> str:

    context_prompt = build_context_prompt(tags)

    system_prompt = baseline_prompt.replace("<tempo>", str(tempo))
    system_prompt = system_prompt.replace("<bar>", str(bars))
    
    proposed_prompt = invoke_claude(context_prompt, system_prompt)

    proposed_prompt = proposed_prompt.replace("\n", "")
    tag = "ia-drummer"
    reg_str = "<" + tag + ">(.*?)</" + tag + ">"
    res = re.findall(reg_str, proposed_prompt)

    if len(res) > 0:
        cleaned_up_prompt = res[0]
    else:
        cleaned_up_prompt = "n/a"

    context.append({
        "audio_analysis_tags": tags,
        "proposed_prompt": cleaned_up_prompt
    })

    return cleaned_up_prompt

if __name__ == "__main__":
    print(get_prompt_from_tags(["Slow", "Rock", "Male"], 75, 8))
    print(get_prompt_from_tags(["slow","ambient","quiet","new age","soft"], 75, 8))
    #get_prompt_from_tags(["Rock", "quiet", "slow"], 75, 8)
    