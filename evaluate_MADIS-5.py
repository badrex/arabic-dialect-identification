import os
os.environ['HF_HOME'] = '/nethome/badr/projects/voice_conversion/hf_models'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/nethome/badr/projects/voice_conversion/wandb'

from transformers import AutoModelForAudioClassification, AutoProcessor
from transformers import AutoFeatureExtractor
from transformers import pipeline
from datasets import load_from_disk
import torch

# calculate accuracy and f1 score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tqdm import trange, tqdm


dir_to_dialect = {
    'EGY': 'Egyptian',
    'GLF': 'Gulf',
    'LEV': 'Levantine',
    'LAV': 'Levantine',
    'MSA': 'MSA',
    'NOR': 'Maghrebi',

    # or identical if already in the same format
    'Egyptian': 'Egyptian',
    'Gulf': 'Gulf',
    'Levantine': 'Levantine',
    'MSA': 'MSA',
    'Maghrebi': 'Maghrebi',
    'Modern Standard Arabic': 'MSA',

}


# # Load model and processor from local directory

main_dir = "/nethome/badr/projects/voice_conversion/"

# fine-tuned models 
models = {
    "top w2vBERT 2.0": "inprogress/adi5-vc-w2v-bert-2.0-5e-06-10-14589-842-vc-n-170625_225005",
    "top MMS model": "inprogress/adi5-vc-mms-300m-5e-05-10-14589-842-vc-n-300125_210327"
} 


for model, model_path in models.items():

    # load the model and processor

    # ADI5_classifier = AutoModelForAudioClassification.from_pretrained(
    #     model_path,
    #     num_labels=5,
    #     #ignore_mismatched_sizes=True,  # to ignore size mismatch errors
    # )    

    #print(ADI5_classifier.config.id2label)
    #continue

    # load model as a pipeline
    print(f"Loading {model} model:  {model_path}")
    ADI5_classifier = pipeline(
        "audio-classification",
        model=model_path,
        device='cuda'
    )


    # change this variabke to True to print the results of each sample
    # useful for debugging 
    verbose = False

    # in-domain evaluation
    adi5_ds = load_from_disk("adi5-aljazeera-arabic-dialects-test")
    in_domain_eval_dataset = adi5_ds["test"]

    # change dialect according to the mapping
    in_domain_eval_dataset = in_domain_eval_dataset.map(
        lambda x: {"dialect": dir_to_dialect[x["dialect"]]}
    )


    print("Evaluating the model on the ADI-5 dataset")

    print("----------------------------------------")
    print("Domain       ", "   Acc.   ", "    F1   ")
    print("----------------------------------------")

    y_true = []
    y_pred = []

    for i, sample in tqdm(enumerate(in_domain_eval_dataset), ncols=100, total=len(in_domain_eval_dataset)):

        # take only the first 60 seconds of the audio
        max_duration_in_seconds = 60
        max_length = max_duration_in_seconds * sample["audio"]["sampling_rate"]

        test_segment = sample["audio"]["array"][:max_length]

        output = ADI5_classifier(test_segment)[0]

        #true_label = dir_to_dialect[sample["dialect"]]
        true_label = sample["dialect"]

        y_true.append(true_label)
        y_pred.append(output["label"])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precison = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')


    print(f"In-domain    {acc:^11.4f}{f1:^11.4f}{precison:^11.4f}{recall:^11.4f}")
    print("----------------------------------------")


    #print("Evaluating the model on the MADI-5 dataset")
    # Load dataset
    madi5_dataset = load_from_disk("MADI-5-Eval-0.2")

    # change dialect according to the mapping  
    out_of_domain_eval_dataset = madi5_dataset.map(
        lambda x: {"dialect": dir_to_dialect[x["dialect"]]}
    )

    # filter out data points that are shorter than 1 second in length 
    # TODO: check the length of the audio files to see if there are empty files

    out_of_domain_eval_dataset = out_of_domain_eval_dataset.filter(
        lambda x: x['length'] > 1
    )


    # print the first sample
    #print(eval_dataset[0])

    # check out the number of samples
    #print(f"Total number of samples: {len(out_of_domain_eval_dataset)}", end='\n\n')

    # change dialect LEV to LAV -- commented for now
    # eval_dataset = eval_dataset.map(
    #     lambda x: {"dialect": "LAV" if x["dialect"] == "LEV" else x["dialect"]}
    # )

    # get domains from the dataset 
    domains = out_of_domain_eval_dataset.unique("domain")


    total_y_true = []
    total_y_pred = []

    for domain in domains:

        # if domain == "Wikipedia":
        #     continue

        domain_dataset = out_of_domain_eval_dataset.filter(lambda x: x["domain"] == domain)

        y_true = []
        y_pred = []

        for i, sample in tqdm(enumerate(domain_dataset), ncols=100, total=len(domain_dataset)):

            output = ADI5_classifier(sample["audio"]["array"])[0]

            #true_label = dir_to_dialect[sample["dialect"]]
            true_label = sample["dialect"]

            y_true.append(true_label)
            y_pred.append(output["label"])

            total_y_true.append(true_label)
            total_y_pred.append(output["label"])

            # check out predictions for each sample
            if verbose:
                print(f"{i+1:<5} "
                    f"{domain:>12} "
                    f"{sample['segment_id']:>25} "
                    f"L: {sample['length']:>5.1f}, "
                    f"P: {output['label']}, "
                    f"T: {true_label}, "
                    f"S: {output['score']:>3.3f}" )

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f'{domain:<12}', end=' ') # {len(ds):<4} samples
        #print(ds)
        print(f"{acc:^11.4f}{f1:^11.4f}")   


    # total micro accuracy
    acc = accuracy_score(total_y_true, total_y_pred)
    f1 = f1_score(total_y_true, total_y_pred, average='weighted')

    print("----------------------------------------")
    print(f"Overall      {acc:^11.4f}{f1:^11.4f}")
    print("----------------------------------------")



