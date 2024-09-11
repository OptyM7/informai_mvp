# informai_mvp
Informai MVP

Informai is a form filling AI tool. 
The MVP will consist of:
  124M LLM model trained on common form data, 
  2 common medical form templates (Patient-GP discussion forms, and bloodwork forms),
  2 form structure documents,
  API for sending form requests/receving filled forms

The 124M LLM model will be finetuned on synthetic data of patient-GP discussions, sourced from a larger finetuned Chat LLM model (Chat-GPT 4o). The data will consist of a discussion transcript, form question and answer. 

The model will be assessed on a test dataset both prior and post finetuning, to assess the model improvement. 

