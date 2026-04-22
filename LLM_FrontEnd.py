#Code for LLM ensemble
#import ollama
from datasets import load_dataset, Features, Value
import numpy as np
from utils import *
from tqdm import tqdm
import torch
import pandas as pd
from synoyms import *
from get_typo import *
from datetime import datetime
from run_bert import run_bert
from runModel_API import run_model
import argparse
import os
import torch
import sys
from get_cosine_similarity import *

def generate_responses(test_data, MODEL_NAME,dataset_name, parameter = 'synonyms'):
    responses = []
    print('Length of the test set is {}'.format(len(test_data)))
    for i, item in tqdm(enumerate(test_data)):
        if i < 10:
            pass
        else:
            break
        #print(item)
        prompt_template = """Answer the following question where the answer should be a single sentence of at least 5 words and at maximum 10 words:
    
        [DOCUMENT]
        """
        try:
            text = item[0]
            label = item[1]
        except:
            text = item["statement"]
            label = item["label"]

        prompt = prompt_template.replace("[DOCUMENT]", text)

        if parameter.lower() == 'synonyms':
            list_responses = []
            list_prob_dist = []
            list_text_edited = []
            for i in range(9):
                #original text without any changes
                if i == 0:
                    prompt = prompt_template.replace("[DOCUMENT]", text)
                    #response, prob_dist = run_bert(prompt,model_name=MODEL_NAME) #Change this to GPT
                    resp = run_model(prompt, temp = 1, topk = 1, parameter = parameter, model_name = MODEL_NAME)
                    list_responses.append(resp['content'])
                    #list_prob_dist.append(prob_dist)
                    list_text_edited.append(text)
                #peturbed text with synonyms and typos. 
                else:
                    output, text_edited = get_random_synonyms(text, sample_size=2)
                    text_edited = apply_typo(text_edited,0.05)
                    prompt = prompt_template.replace("[DOCUMENT]", text_edited)
                    #response, prob_dist = run_bert(prompt,model_name=MODEL_NAME) #change this to GPT
                    resp = run_model(prompt, temp = 1, topk = 1, parameter = parameter, model_name = MODEL_NAME)
                    list_responses.append(resp['content'])
                    #list_prob_dist.append(prob_dist)
                    list_text_edited.append(text_edited)
        else:
            raise ValueError('The parameter {} is not recognizable it should be either temp or topk'.format(parameter))

           
        # Create weighted ensemble
        response = {}
        #weights = [w/sum(weights) for w in weights] # Normalize weights
        #majority_voting = np.sum(list_prob_dist, axis = 0) / len(prob_dist_default) #Make the number of the models (trials as input argument)
        #median = np.median(list_prob_dist,axis = 0)
        #print(majority_voting)
        #confidence = kl_ensemble_confidence(list_prob_dist)
        #prediction = np.argmax(majority_voting)
        #prediction_median = np.argmax(median)
        #prediction_default = np.argmax(prob_dist)

        #get cosine similarity between the answer and the LLM response. 

        cosine_similarity_scores = [get_cosine_similarity(resp, label) for resp in list_responses]
        median = np.median(cosine_similarity_scores,axis=0)
        #default_accuracy = (label.lower == list_responses[0].lower())
        default_accuracy = (label.lower() in list_responses[0].lower())*100
        # Get the accuracy between the answer (label) and the responses
        #correct_count = sum(1 for resp in list_responses if resp.lower() == label.lower())
        #accuracy = correct_count / len(list_responses)
        hallucination_metrics = compute_hallucination_metrics(list_responses, reference_answer=label)
        center_response = hallucination_metrics['center_response']
        #accuracy = (center_response.lower() == label.lower())
        #print(center_response)
        accuracy = (label.lower() in center_response.lower())*100

        response = {'sentence': text,
                    'text_edited': list_text_edited,
                    'default_accuracy': default_accuracy,
                    'LLM_response': list_responses,
                    'median_cosine': median,
                    'center_response': center_response,
                    'center_similarity_mean': hallucination_metrics['center_similarity_mean'],
                    'clean_to_center_similarity': hallucination_metrics['clean_to_center_similarity'],
                    'cluster_dispersion': hallucination_metrics['cluster_dispersion'],
                    'reference_similarity': hallucination_metrics['reference_similarity'],
                    'hallucination_score': hallucination_metrics['hallucination_score'],
                    'label_gt': label,
                    'accuracy': accuracy
                    # 'label_pred_majority_voting': prediction,
                    # 'kl_confidence':confidence
                    }
        responses.append(response)
    return responses



def load_process_data(dataset_name,API_Key,MODEL_NAME,save_results_dir):
    #dataset_name = 'rotten_tomatoes'#'stanfordnlp/imdb' #"rotten_tomatoes"
    if dataset_name.lower() == 'rotten tomatoes':
        dataset_name = 'rotten_tomatoes'
        split_flag = 'train_test'
    elif dataset_name.lower() == 'imdb':
        dataset_name = 'stanfordnlp/imdb'
        split_flag = 'train_test'
    elif dataset_name.lower() == 'pminervini/true-false':
        dataset_name = 'pminervini/true-false'
        split_flag = 'facts'
    elif dataset_name.lower() == 'iamasq/defan':
        dataset_name = 'iamasQ/DefAn'
        split_flag = 'train'
    else:
        raise ValueError('Error in the name of the dataset, do python LLM_FrontEnd.py --help')
    
    try:
        data = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset with default settings: {e}")
        print("Attempting to load with alternative method...")
        import json
        import pandas as pd

        with open("./QA_domain_1_public.json", "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Force problematic column to string
        df["answer"] = df["answer"].astype(str)
        #print(df.head(5))
        #sys.exit()
    
    #if split_flag == 'train':
    #    test_data = data["train"]
    try:
        test_data = data[split_flag]
    except:
        test_data = df.to_numpy()
    # Shuffle the test dataset
    #test_data = test_data.shuffle(seed=42)

    print("In this experiment we are using the test split for evaluation from dataset {} ".format(dataset_name))
    print('Test set size {} '.format(test_data.shape[0]))
    print('An example from the dataset is below \n {} '.format(test_data[1]))
    #print('Unique labels in the test set are {} '.format(np.unique(test_data['answer'])))

    #MODEL_NAME ='distilbert-base-uncased-finetuned-sst-2-english'#'openai/gpt-oss-20b'
    parameter = 'synonyms' 
    print("Now trying {}".format(parameter))
    output = generate_responses(test_data,MODEL_NAME, dataset_name, parameter)
    df = pd.DataFrame(output)

    # Ensure columns are numeric
    #df['label_gt'] = df['label_gt'].astype(str)
    #f['Response'] = df['LLM_response'].astype(str)
    #df['label_pred_majority_voting'] = df['label_pred_majority_voting'].astype(int)
    #df['kl_confidence'] = df['kl_confidence'].astype(float)
    #accuracy_default = (df['label_gt'] == df['label_pred_default']).mean()
    #accuracy_majority = (df['label_gt'] == df['label_pred_majority_voting']).mean()
    #average_confidence = df['kl_confidence'].mean()
    accuracy = df['accuracy'].mean()
    accuracy_default = df['default_accuracy'].mean()
    cosine_similarity = df['median_cosine'].mean()
    hallucination_score = df['hallucination_score'].mean()
    reference_similarity = df['reference_similarity'].mean()
    center_similarity = df['center_similarity_mean'].mean()

    print(f'Accuracy (label_pred vs label_gt): {accuracy:.4f}')
    print(f'Average of mean cosine similarity is ): {cosine_similarity:.4f}')
    print(f'Average of accuracy for the default prompt is ): {accuracy_default:.4f}')
    print(f'Average hallucination score: {hallucination_score:.4f}')
    print(f'Average center/reference similarity: {reference_similarity:.4f}')
    print(f'Average response consistency: {center_similarity:.4f}')
    
    # Save metrics to text file
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(save_results_dir, f'{dataset_name.replace("/","")}_metrics_{timestamp}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f'Accuracy : {accuracy:.4f}\n')
        f.write(f'Accuracy default prompt: {accuracy_default:.4f}\n')
        f.write(f'Average of mean cosine similarity: {cosine_similarity:.4f}\n')
        f.write(f'Average hallucination score: {hallucination_score:.4f}\n')
        f.write(f'Average center/reference similarity: {reference_similarity:.4f}\n')
        f.write(f'Average response consistency: {center_similarity:.4f}\n')
    df.to_csv(os.path.join(save_results_dir,dataset_name.replace('/','')+parameter+'_responses_output_synonyms_EntireTestSet'+'_MerriamWebster_'+
                           MODEL_NAME.replace('/','')+f'_{timestamp}.csv'), index=False)
    
def main():
    parser = argparse.ArgumentParser(description='Argument for code to LLM Ensemble')
    parser.add_argument('--dataset',type=str,default='iamasq/defan', required=False,help='Name of the dataset, which can be iamasQ/DefAn')
    parser.add_argument('--API_Key',type=str,required=False,default='../API_Key/MerriamWebster.txt',help='path to API_Key for Merriam Webster https://dictionaryapi.com/')
    parser.add_argument('--save_results',type=str,required=False,default='./results',help='path to save the results of the LLM FrontEnd')
    parser.add_argument('--LLM',type=str,required=False,default="phi4:latest",help='LLM for getting the predictions, default is GPT-5-nano')
    args = parser.parse_args()
    print('torch version: {} , cuda available: {}'.format(torch.__version__,torch.cuda.is_available()))
    load_process_data(args.dataset, args.API_Key, args.LLM, args.save_results)
    print('Done')

if __name__ == '__main__':
    main()
