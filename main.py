import re
import os
import wandb
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


def preprocess_text(text):
    if text is None:
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s,.!?\"'():;-]", "", text)  # Remove non-standard characters
    text = re.sub(r"[\-_]", " ", text)  # Replace hyphens and underscores with spaces
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\\\", "")
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


def load_and_preprocess_data(in_file, expected_file):
    """
    Loads input and expected files, preprocesses them, and creates a Hugging Face Dataset.
    """
    # Load the TSV files
    in_data = pd.read_csv(in_file, sep="\t", header=None)  # No header
    expected_data = pd.read_csv(expected_file, sep="\t", header=None, names=["output_text"])  # Add column name

    # Extract the 4th column as input_text
    in_data["input_text"] = in_data.iloc[:, 3]  # Use the 4th column (index 3)

    # Combine the datasets into a single DataFrame
    combined_data = pd.concat([in_data["input_text"], expected_data["output_text"]], axis=1)

    # Convert the DataFrame into a Dataset object
    hf_dataset = Dataset.from_pandas(combined_data)

    # Apply preprocessing
    hf_dataset = hf_dataset.map(
        lambda examples: {
            "input_text": preprocess_text(examples["input_text"]),
            "output_text": preprocess_text(examples["output_text"])
        }
    )
    return hf_dataset


def train_model(tokenized_dataset, tokenizer, model_name="t5-small"):
    """
    Fine-tunes the T5 model for text correction.
    """
    # Split the dataset into train and test sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # Load the T5 model for conditional generation
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir="./results",  # Directory to save model checkpoints
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,  # Adjusted for smaller models
        num_train_epochs=3,  # Reduced epochs for faster training
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=None
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    # Evaluate the model
    evaluation_results = trainer.evaluate()
    print(f"Evaluation results: {evaluation_results}")


def load_fine_tuned_model():
    """
    Loads the fine-tuned model and tokenizer.
    """
    model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_model")
    tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_model")
    return model, tokenizer


def correct_sentence(sentence, tokenizer, model):
    """
    Corrects a sentence using the fine-tuned model.
    """
    input_text = f"Popraw ten tekst: {sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=256, no_repeat_ngram_size=2, num_beams=4)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence


def tokenize_function(examples):
    inputs = tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    outputs = tokenizer(
        examples["output_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs


if __name__ == '__main__':
    # Requires api key from wandb
    load_dotenv()
    api_key = os.getenv("API_KEY")
    wandb.login(key=api_key)

    # File paths
    in_file = "./train/in.tsv"
    expected_file = "./train/expected.tsv"

    # Load tokenizer and data
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    hf_dataset = load_and_preprocess_data(in_file, expected_file)

    for i in range(5):
        print(f"Input: {hf_dataset[i]['input_text']}")
        print(f"Output: {hf_dataset[i]['output_text']}\n")

    # Tokenize dataset
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True
    )

    print(tokenized_dataset[0])

    # Train the model if not already trained
    if not os.path.exists("./fine_tuned_model"):
        train_model(tokenized_dataset, tokenizer)

    # Load the fine-tuned model
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()

    # Example usage
    test_sentence = "ca Wtem\n\nwśród postów i ciężkich umartwień, nad\ngrobem własną ręką wykopanym, w nieu-\nstannem rozmyślaniu o śmierci, czekać tej\nśmierci, jako wyzwolenia z pęt ziemskich.\n\nJuż starania poczynił, aby go do tego\nzgromadzenia przyjęto, zezwolenie wladz\nwłoskich uzyskał, wszelkie formalności za-\nłatwił, już miała się zamknąć za nim na\nzawsze ciężka furta, gdy zaszła szczęśliwa\nokoliczność, która jego zamiarom, nie zmie-\nniając ich w zasadzie, uadała inny kie-\nranek.\n\nSpotkał pod włoskiem niebem ziomka ka-\npłana, człowieka o sercu zacnem i umyśle\njasnym i zwierzył mu się ze swoich inten-\ntyj. Ten, wysłachawszy młodzieńca, nie\nodwodził go od powziętego postanowienia,\nowszem, vmacniał go w niem i słów zacltę-\ncy nie szczędził.\n\n— Jdź—mówił—za głosem, który cię\nwoła, wyrzecz się świata, służ Bogu — ale\nnie tutaj jest twoje miejsce. Wracaj tam,\nskąd przyszedłeś, do swoich. Wstąp do\nzgromadzenia żebrzących i pokorrych, służ\n\nhttp://rcin.org.pl\n"
    corrected_sentence = correct_sentence(test_sentence, fine_tuned_tokenizer, fine_tuned_model)
    print(f"Original Sentence: {test_sentence}")
    print(f"Corrected Sentence: {corrected_sentence}")
