import re
import pandas as pd
import argparse


def extract_links_to_excel(input_txt_file, output_excel_file):
    """
    Extracts all links from a .txt file and writes them into an Excel file.

    Parameters:
        input_txt_file (str): Path to the input .txt file.
        output_excel_file (str): Path to the output Excel file.
    """
    try:
        # Read the contents of the .txt file
        with open(input_txt_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regular expression to find URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = re.findall(url_pattern, content)

        if not links:
            print("No links found in the file.")
            return

        # Create a DataFrame to store the links
        df = pd.DataFrame(links, columns=["Links"])

        # Write the DataFrame to an Excel file
        df.to_excel(output_excel_file, index=False, engine='openpyxl')

        print(f"Links have been successfully extracted to '{output_excel_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_txt_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract links from a text file and save them to an Excel file.")
    parser.add_argument("input_txt_file", type=str, help="Path to the input .txt file.")
    parser.add_argument("output_excel_file", type=str, help="Path to the output Excel file.")

    args = parser.parse_args()

    # Call the function with parsed arguments
    extract_links_to_excel(args.input_txt_file, args.output_excel_file)
