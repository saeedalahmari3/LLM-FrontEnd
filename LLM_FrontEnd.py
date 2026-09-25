from category_metrics import is_mbzuai, normalize_category, compute_category_accuracy
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
from runModel_API import translateSentence

def generate_responses(test_data, MODEL_NAME,dataset_name, parameter = 'synonyms'):
    responses = []
    print('Length of the test set is {}'.format(len(test_data)))
    print(test_data.head())
    for i, (_,item) in tqdm(enumerate(test_data.iterrows()),total=len(test_data)):
        #if i < 10:
        #    pass
        #else:
        #   break
        #print(item)
        # what sentiment is this textual review write either positive or negative then provide a brief reason with a maximum of five words:
        #Do not search internet. Answer the following question where the answer should be at least 5 words and at most 20 words, show the steps in brief and at the end show the final numerical
        #answer proceeded with three hashtags ###:
        #Do not search internet. Answer the following question where the answer should be a single sentence of at least 5 words and at maximum 10 words:
        #Answer the following question in 5 words, if you don't know the answer then response only by 'I don't have an answer for this question'.
        # """ Answer the following question by choosing only one from the following four options then provide 5 words ONLY justifying your choice'.
        #Question: [DOCUMENT]

        #Options: 1. did_not_happen
        #         2. far_future
        #         3. nonsense
        #         4. obscure 

        #defintions: did_not_happen: means question asking about something did not happen at all in the past, far_future: means the question is asking about 
        #            something in the future, nonsense: the question do not make sense, obscure: the answer to the question is unknown. 
        #"""
        
        prompt_template = """ Answer the following question by True if you think the question answer is either did_not_happen, far_future, nonsense, obscure or correct.  Otherwise answer the question by False. In both cases provide a maximum of 5 words justification'.
        Question: [DOCUMENT]
        """
        try:
            text =  item['question'] # item[0]
            label = item['category'] # item[1]
            #print(text)
            #print(label)
        except:
            #print('here')
            text = item['question'] #text = item["Question"]
            #label = item['label']  #label = item["Best Answer"]
            #label = " I don't have an answer for this question"
            label = item['category']
            
        if is_mbzuai(dataset_name):
            label = normalize_category(item['category'], i + 2)
        label = str(label)

        prompt = prompt_template.replace("[DOCUMENT]", text)

        if parameter.lower() == 'synonyms':
            list_responses = []
            list_prob_dist = []
            list_text_edited = []
            for i in range(19):
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
                    #output, text_edited = get_random_synonyms(text, sample_size=2)
                    text_edited = apply_typo(text,0.09)
                    #print(text)
                    #text_edited = translateSentence(text,prompt_task="Translate the following English sentence to Arabic, where the response should be Arabic text only:")
                    #print('translation to Arabic {}'.format(text_edited))
                    #text_edited = translateSentence(text_edited,prompt_task="Translate the following Arabic sentence to English, where the response should be English text only:")
                    #print('translation to English {}'.format(text_edited))
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

        cosine_similarity_scores = [get_cosine_similarity(resp, label) for resp in list_responses]
        median = np.median(cosine_similarity_scores,axis=0)
        hallucination_metrics = compute_hallucination_metrics(list_responses, label, reference_answer=label)
        center_response = hallucination_metrics['center_response']


        response = {'sentence': text,
                    'text_edited': list_text_edited,
                    'LLM_response': list_responses,
                    'median_cosine': median,
                    'center_response': center_response,
                    'center_similarity_mean': hallucination_metrics['center_similarity_mean'],
                    'clean_to_center_similarity': hallucination_metrics['clean_to_center_similarity'],
                    'hallucination_score_clean_vs_label': hallucination_metrics['hallucination_score_clean_vs_label'],
                    'hallucination_score_center_vs_label': hallucination_metrics['hallucination_score_center_vs_label'],
                    'hallucination_score_center_vs_clean': hallucination_metrics['hallucination_score_center_vs_clean'],
                    'label_gt': label
                    }
        if is_mbzuai(dataset_name):
            response['dataset_name'] = dataset_name
            response['category'] = label
            response['accuracy_clean_vs_label'] = compute_category_accuracy(
                label, list_responses[0]
            )
            response['accuracy_center_vs_label'] = compute_category_accuracy(
                label, center_response
            )
        responses.append(response)
    return responses



def load_process_data(dataset_name,API_Key,MODEL_NAME,save_results_dir):
    #dataset_name = 'rotten_tomatoes'#'stanfordnlp/imdb' #"rotten_tomatoes"
    if dataset_name.lower() == 'rotten tomatoes':
        dataset_name = 'rotten_tomatoes'
        split_flag = 'train_test'
    elif dataset_name.lower() == 'imdb':
        dataset_name = 'stanfordnlp/imdb'
        split_flag = 'test'
    elif dataset_name.lower() == 'pminervini/true-false':
        dataset_name = 'pminervini/true-false'
        split_flag = 'facts'
    elif dataset_name == 'domenicrosati/TruthfulQA':
        dataset_name = 'domenicrosati/TruthfulQA'
        split_flag = 'train'
    elif dataset_name.lower() == 'iamasq/defan':
        dataset_name = 'iamasQ/DefAn'
        split_flag = 'train'
    elif dataset_name.lower() == 'openai/gsm8k':
        dataset_name = 'openai/gsm8k'
        split_flag = 'test'
    elif dataset_name.lower() == 'mbzuai':
        dataset_name = 'MBZUAI/LaMini-Hallucination'
        split_flag = 'test'
    elif dataset_name.lower() == 'mbzuai_expanded':
        dataset_name = '/Users/saeedalahmari/Documents/LLM_ensemble_USF/code/LLMFrontEnd/mbzuai_extended_with_correct.csv'
        split_flag = 'test'
    else:
        raise ValueError('Error in the name of the dataset, do python LLM_FrontEnd.py --help')

    try:
        data = load_dataset(dataset_name)
    except Exception as e:
        #print(f"Error loading dataset with default settings: {e}")
        #print("Attempting to load with alternative method...")
        #import json
        import pandas as pd

        #with open("./QA_domain_1_public.json", "r") as f:
        #    data = json.load(f)

        df = pd.read_csv(dataset_name)

        # Force problematic column to string
        #df["answer"] = df["answer"].astype(str)
        #print(df.head(5))
        #sys.exit()
    
    #if split_flag == 'train':
    #    test_data = data["train"]
    try:
       test_data = data[split_flag]
    except:
        test_data = df
    # Shuffle the test dataset
    #test_data = test_data.shuffle(seed=42)

    print("In this experiment we are using the test split for evaluation from dataset {} ".format(dataset_name))
    print('Test set size {} '.format(test_data.shape[0]))
    #print('An example from the dataset is below \n {} '.format(test_data[1]))
    #print('Unique labels in the test set are {} '.format(np.unique(test_data['answer'])))

    #MODEL_NAME ='distilbert-base-uncased-finetuned-sst-2-english'#'openai/gpt-oss-20b'
    parameter = 'synonyms' 
    print("Now trying {}".format(parameter))
    output = generate_responses(test_data,MODEL_NAME, dataset_name, parameter)
    import pandas as pd
    df = pd.DataFrame(output)

    cosine_similarity = df['median_cosine'].mean()
    hallucination_h = df['hallucination_score_clean_vs_label'].mean()
    hallucination_h_labelvsCenter = df['hallucination_score_center_vs_label'].mean()
    hallucination_h_cleanvsCenter = df['hallucination_score_center_vs_clean'].mean()
    center_similarity = df['center_similarity_mean'].mean()
    print(f'Average of mean cosine similarity is ): {cosine_similarity:.4f}')
    #print(f'Average of accuracy for the default prompt is ): {accuracy_default:.4f}')
    print(f'Average hallucination score: {hallucination_h:.4f}')
    print(f'Average hallucination score for Center vs. Label: {hallucination_h_labelvsCenter:.4f}')
    print(f'Average hallucination score for Center vs. Clean Response: {hallucination_h_cleanvsCenter:.4f}')
    print(f'Average response consistency: {center_similarity:.4f}')
    
    accuracy_summary = []
    if is_mbzuai(dataset_name):
        for column in ('accuracy_clean_vs_label', 'accuracy_center_vs_label'):
            line = (
                f'{column}: average={df[column].mean():.6f}, '
                f'std={df[column].std(ddof=0):.6f}'
            )
            accuracy_summary.append(line)
            print(line)

    # Save metrics to text file
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(save_results_dir, f'{dataset_name.replace("/","")}_metrics_{timestamp}.txt')
    with open(metrics_file, 'w') as f:
        for line in accuracy_summary:
            f.write(line + '\n')
        #f.write(f'Accuracy default prompt: {accuracy_default:.4f}\n')
        f.write(f'Average of mean cosine similarity: {cosine_similarity:.4f}\n')
        f.write(f'Average hallucination score: {hallucination_h:.4f}\n')
        f.write(f'Average hallucination score for Center vs. Label: {hallucination_h_labelvsCenter:.4f}\n')
        f.write(f'Average hallucination score for Center vs. Clean Response: {hallucination_h_cleanvsCenter:.4f}\n')
        f.write(f'Average response consistency: {center_similarity:.4f}\n')
    df.to_csv(os.path.join(save_results_dir,dataset_name.replace('/','')+parameter+'_responses_output_synonyms_EntireTestSet'+'_MerriamWebster_'+
                           MODEL_NAME.replace('/','')+f'_{timestamp}.csv'), index=False)
    
def main():
    parser = argparse.ArgumentParser(description='Argument for code to LLM Ensemble')
    parser.add_argument('--dataset',type=str,default='mbzuai_expanded', required=False,help='Name of the dataset, which can be iamasQ/DefAn')
    parser.add_argument('--API_Key',type=str,required=False,default='../API_Key/MerriamWebster.txt',help='path to API_Key for Merriam Webster https://dictionaryapi.com/')
    parser.add_argument('--save_results',type=str,required=False,default='../results',help='path to save the results of the LLM FrontEnd')
    parser.add_argument('--LLM',type=str,required=False,default="gemma3:12b",help='LLM for getting the predictions, default is GPT-5-nano')
    args = parser.parse_args()
    print('torch version: {} , cuda available: {}'.format(torch.__version__,torch.cuda.is_available()))
    load_process_data(args.dataset, args.API_Key, args.LLM, args.save_results)
    print('Done')

if __name__ == '__main__':
    main()
