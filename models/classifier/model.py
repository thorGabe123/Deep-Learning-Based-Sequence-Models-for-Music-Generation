from transformers import BertForSequenceClassification, BertConfig

num_labels = 5  # Change to your number of labels

config = BertConfig(
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

model = BertForSequenceClassification(config)