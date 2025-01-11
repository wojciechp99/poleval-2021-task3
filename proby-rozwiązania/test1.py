import csv
import difflib
import re

import jiwer
import language_tool_python
import pandas as pd
from autocorrect import Speller
from pandas import DataFrame
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def calculate_overall_wer(reference_texts, hypothesis_texts):
    """
    Calculate overall Word Error Rate (WER) for all rows combined.
    """
    # Combine all reference and hypothesis texts into single strings
    combined_reference = " ".join(reference_texts)
    combined_hypothesis = " ".join(hypothesis_texts)

    # Calculate WER using jiwer
    return jiwer.wer(combined_reference, combined_hypothesis)


def clean_text(raw_text):
    # Step 1: Remove \n and \\ (escape sequences)
    cleaned_text = raw_text.replace("\\n", " ").replace("\\\\", "")

    # Step 2: Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def correct_text_nlp(text):
    # use the autocorrect library
    spell = Speller(lang="pl", only_replacements=True)
    corrected_test = spell(text)
    return corrected_test


def use_autocorrection_library(cleaned_text, reference_texts):
    # Apply correct_text_nlp only on the first 10 rows
    corrected_text = cleaned_text.copy()  # Copy the cleaned_text for modification
    corrected_text.iloc[:10] = corrected_text.iloc[:10].apply(correct_text_nlp)

    print("Corrected Text using autocorrection library (First 10 Rows):")
    print(corrected_text.head(10))

    # Ensure lengths match for reference and hypothesis
    corrected_first_10 = corrected_text.iloc[:10].tolist()
    if len(reference_texts) != len(corrected_first_10):
        raise ValueError("Number of reference texts must match number of rows in hypothesis.")

    # Compute overall WER
    overall_wer = calculate_overall_wer(reference_texts, corrected_first_10)
    print(f"Overall WER for first 10 rows: {overall_wer}")

    # Compute differences for the first 10 rows
    print("\nDifferences between cleaned and corrected text (First 10 Rows):\n")
    for i, (clean_row, correct_row) in enumerate(zip(cleaned_text.iloc[:10], corrected_first_10)):
        diff = list(difflib.unified_diff(
            clean_row.split(),
            correct_row.split(),
            lineterm="",
            fromfile="cleaned_text",
            tofile="corrected_text"
        ))
        print(f"Row {i + 1} Diff:")
        print("\n".join(diff))
        print("-" * 50)


def use_language_tool_python(text_df, reference_texts):
    # Get the first 10 rows of the DataFrame column
    text_list = text_df.iloc[:10].squeeze().tolist()

    # Initialize the language tool for Polish
    tool = language_tool_python.LanguageTool('pl')

    # Apply corrections row by row
    corrected_texts = []
    for original_text in text_list:
        matches = tool.check(original_text)
        corrected_text = language_tool_python.utils.correct(original_text, matches)
        corrected_texts.append(corrected_text)

    result_df = DataFrame(corrected_texts)
    print(result_df)

    # Compute overall WER
    overall_wer = calculate_overall_wer(reference_texts, corrected_texts)
    print(f"Overall WER for first 10 rows: {overall_wer}")


def use_pretrained_model(text, reference_texts):
    # Load model and tokenizer
    model_name = "dkleczek/bert-base-polish-uncased-v1"
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Create a fill-mask pipeline
    nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    def correct_polish_text(text):
        tokens = text.split()  # Split text into tokens
        corrected_tokens = []

        # Iterate through tokens and correct each one
        for i, token in enumerate(tokens):
            # Replace the current token with [MASK]
            masked_text = " ".join(tokens[:i] + ["[MASK]"] + tokens[i + 1:])

            # Use the pipeline to predict the masked token
            predictions = nlp(masked_text)

            # Get the top prediction
            corrected_word = predictions[0]["token_str"]
            corrected_tokens.append(corrected_word)

        # Join corrected tokens into a sentence
        corrected_text = " ".join(corrected_tokens)
        return corrected_text

    corrected_text = correct_polish_text(text)

    print("Original Text:", text)
    print("Corrected Text:", corrected_text)


if __name__ == '__main__':
    file_path = "./train/in.tsv"
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            data.append({'doc_id': row[0], 'page_num': row[1], 'year': row[2], 'ocr_text': row[3]})

        df = pd.DataFrame(data)
        pd.set_option('display.max_colwidth', 700)

        # Clean the text
        text = df['ocr_text']
        cleaned_text = text.apply(clean_text)
        print("Cleaned Text (First 10 Rows):")
        print(cleaned_text.head(10))

        # Load reference file (expected.tsv)
        reference_data = []
        with open("./train/expected.tsv", mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                reference_data.append(row[0])

        # Use the first 10 rows from the reference file
        reference_texts = reference_data[:10]

        # Apply clean_text function to each reference text
        cleaned_reference_texts = [clean_text(text) for text in reference_texts]

        # use_autocorrection_library(cleaned_text, cleaned_reference_texts)

        # use_language_tool_python(cleaned_text, cleaned_reference_texts)

        # use_pretrained_model(cleaned_text[0], cleaned_reference_texts)