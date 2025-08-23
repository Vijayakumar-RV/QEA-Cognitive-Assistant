import json
import pandas as pd

def markdown_testcase_from_json(json_string):
    if isinstance(json_string, str):
        print(f"Parsing JSON string: Came inside markdown_testcase_from_json with string input")
        test_cases = json.loads(json_string)
    else:
        print(f"since not json Parsing JSON object: Came inside markdown_testcase_from_json with object input")
        test_cases = json_string

    print(f"Test Cases from response: {test_cases}")

    # Generate Markdown format for each test case
    for tc in test_cases:
        markdown_str = f"""### Test Case ID: {tc["Test Case ID"]}
    ##Title##: {tc["Test Case Title"]}  
    ##Type##: {tc["Test Type"]}  
    ##Priority##: {tc["Priority"]}  
    ##Tags##: {", ".join(tc["Tags"])}  

    ##Description##:  
    {tc["Description"]}

    ##Preconditions##:
    """ 
        if isinstance(tc["Preconditions"], list):
            markdown_str += "\n".join(f"- {pre}" for pre in tc["Preconditions"]) + "\n\n"
        else:
            markdown_str += f"- {tc['Preconditions']}\n\n"

        markdown_str += "##Test Steps##:\n" + "\n".join(f"{step}" for step in tc["Test Steps"]) + "\n\n"
        markdown_str += f"##Expected Result##:\n{tc['Expected Result']}\n"
    print(f"Markdown Test Case: {markdown_str}")
    return markdown_str


def save_testcase_to_csv(test_cases, file_path):
    """
    Save test cases to a CSV file.
    
    Args:
        test_cases (list): List of test case dictionaries.
        file_path (str): Path to save the CSV file.
    """
    # Convert basic details to a dataframe
    summary_data = [{
        "Test Case ID": tc["Test Case ID"],
        "Title": tc["Test Case Title"],
        
        "Type": tc["Test Type"],
        "Priority": tc["Priority"],
        "Tags": ", ".join(tc["Tags"]),
        "Test Steps": "\n".join(tc["Test Steps"])  # You can also use "; ".join(...) if preferred
    } for tc in test_cases]
    
    df = pd.DataFrame(summary_data)
    df.to_csv(file_path, index=False)
    print(f"Test cases saved to {file_path}")