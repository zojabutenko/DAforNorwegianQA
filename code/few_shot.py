from argparse import ArgumentParser
import torch
import os, re
import gzip, json
import tarfile
import numpy as np
import random
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import time


torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)  # Print memory usage info
CUDA_LAUNCH_BLOCKING="1"


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_norquad_contexts(norquad_dir):
    # load norquad to check that we don't generate with articles that are already in the dataset
    contexts = set()
    for file_name in ['training_dataset_flattened.json', 'test_dataset_flattened.json']:
        file_path = os.path.join(norquad_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data['data']:
                for paragraph in entry['paragraphs']:
                    contexts.add(paragraph['context'])
    return contexts


def load_dataset(file_path):
    # Load dataset and yield instances one at a time
    if file_path.endswith('newspaper_ocr_no.txt.gz'):
        with gzip.open(file_path,'rt', encoding='utf-8') as fin:
            for line in fin:
                yield line.strip()

    # load wiki
    elif file_path.endswith('wikipedia.tar.gz'):
        yield from load_wiki(file_path, 'nob.wikipedia.json', '/cluster/home/zoiab/thesis/NorQuAD/data/evaluation/all')
    
    # load norquad test set for evaluation with n-gram-based scores
    elif file_path.endswith('test_dataset_flattened.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data['data']:
                for paragraph in entry['paragraphs']:
                    yield paragraph['context']

def load_wiki(archive_path, target_file, norquad_dir):
    # Load NorQuAD contexts
    norquad_contexts = load_norquad_contexts(norquad_dir)
    data_path = '/cluster/projects/nn9851k/zoiab/data/'

    # check if the target file is already extracted
    if os.path.exists(data_path + target_file):
        pass
    else:
        # Extract the target file from the archive
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extract(target_file, path=data_path)

    # Load the JSON data from the extracted file and yield each article text
    with open(data_path + target_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
        for article in articles:
            text = article['text']
            words = re.findall(r'\w+', text)
            # set length for the articles to around 300-400 words as normal for NorQuAD wiki subset
            if len(words) >= 300 and len(words) > 400:
                truncated_text = ' '.join(words[:400])
                if truncated_text not in norquad_contexts:
                    yield truncated_text
            elif len(words) >= 300 and len(words) <= 400:
                if text not in norquad_contexts:
                    yield text

def format_qas(questions, answers, article, text_id=1):
    # format generated question-answer pair as JSON for easier access
    qas_list = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
    output = {
        "text_id": text_id,
        "text": article,
        "qas": qas_list
    }
    return json.dumps(output, ensure_ascii=False)


def extract_qas(text):
    qa_pairs = re.findall(r'(\d+)\.\s*Spørsmål:\s*(.*?)\n\s*Svar:\s*(.*)', text)
    questions = [q.strip() for _, q, _ in qa_pairs]
    answers = [a.strip() for _, _, a in qa_pairs]
    answers = [a.strip("<|im_end|>") for _, _, a in qa_pairs]
    answers = [re.sub(r'</s>.*', '', a) for a in answers]
    return questions, answers


@torch.no_grad()
def zero_shot_normistral(text, tokenizer, model, device):
    input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
    eos_token = '/s'
    prediction = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,  
        top_p=0.9,         
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    )

    # decoded = tokenizer.batch_decode(prediction)
    # pred = decoded[0]
    # return pred[len(text):]
    return tokenizer.decode(prediction[0, input_ids.size(1):]).strip()



@torch.no_grad()
def zero_shot_instruct_normistral(text, tokenizer, model, device):    
    encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
    model_inputs = encodeds.to(device)
    eos_token = '/s'
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=512,
        top_k=32,  # top-k sampling
        # top_p=0.9,  # nucleus sampling
        temperature=0.4,  # a low temparature to make the outputs less chaotic
        repetition_penalty=1.0,  # turn the repetition penalty off, having it on can lead to very bad outputs
        do_sample=True,  # randomly sample the outputs
        use_cache=True,  # speed-up generation
        eos_token_id=tokenizer(eos_token).input_ids,
        bos_token_id=0,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.batch_decode(generated_ids)
    pred = decoded[0]
    return pred[len(text):]


@torch.no_grad()
def zero_shot_instruct(text, tokenizer, model, device):    
    encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to(device)
    eos_token = '/s'
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=512, 
        top_k=32,  # top-k sampling
        do_sample=True,  # randomly sample the outputs
        use_cache=True,  # speed-up generation
        eos_token_id=tokenizer(eos_token).input_ids,
        bos_token_id=0,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.batch_decode(generated_ids)
    pred = decoded[0]
    return pred[len(text):]



def main():
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )

    parser = ArgumentParser()
    parser.add_argument("--datapath", default="/cluster/projects/nn9851k/zoiab/newspaper_ocr_no.txt.gz")
    parser.add_argument("--modelpath", default="/cluster/projects/nn9851k/models/")
    parser.add_argument("--prompt", default="") # number of prompt to add to output filename
    parser.add_argument("--model", default="Mistral-7B-Instruct-v0.2") # possible values: Mistral-7B-Instruct-v0.2, norallm/normistral-7b-warm-instruct, norallm/normistral-11b-warm
    parser.add_argument("--outpath", default="/cluster/projects/nn9851k/zoiab/data/few-shot/")
    parser.add_argument("--fewshot", action='store_true')
    parser.add_argument("--run_from", default=0) # if we want to continue from where we left of, we can choose id of text in dataset, e.g. run_from=1400 will start from instance 1400 in dataset
    parser.add_argument("--num_instances", default=100) # how many articles to process


    parser.add_argument("--seed", action="store", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    
    logging.info(args)
    
    if args.model == 'Mistral-7B-Instruct-v0.2':
        model_name = args.modelpath + args.model
    elif args.model == 'norallm/normistral-7b-warm-instruct' or args.model=='norallm/normistral-11b-warm':
        model_name = args.model


    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/cluster/projects/nn9851k/zoiab/few-shot/cache"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="/cluster/projects/nn9851k/zoiab/few-shot/cache",
        torch_dtype=torch.float16,  # Use fp16 instead of bfloat16
        device_map="auto",  # Auto-allocate model layers to available memory
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True)  # Load in 8-bit
    ).eval()
    
    data_subset = 'news' if args.datapath.endswith('newspaper_ocr_no.txt.gz') else 'wiki'

    if args.datapath.endswith('test_dataset_flattened.json'):
        filename = 'test_predictions'
    else:

        filename = f'{args.model[:3]}{args.model[18:22]}_{data_subset}_from{args.run_from}'

    logging.info('Loaded the model')
    if args.fewshot:
        prompt = """Du må stille så mange forskjellige HV-spørsmål (hva, hvor, hvem, når, osv.) som mulig til teksten. Alle spørsmålene må ha et riktig svar i teksten som er kort og konsist, for eksempel en frase. Svaret må tas fra teksten akkurat som det står! Du kan IKKE omformulere svaret eller legge til dine egne ord.

        Eksempel:
        Teksten:
        Norske OL-gullvinnere til sykehus etter sykkelulykke\nSverre Lunde Pedersen (t.v.) og Simen Spieler Nilsen er to av Norges beste skøyteløpere. Foto: Lise Åserud / NTB scanpix\nDe to gikk sammen på det norske laget som vant gull på lagtempo i Pyeongchang-OL i fjor. Sverre Lunde Pedersen tok også bronse på 5000 meter i samme mesterskap. Han har i tillegg vunnet flere medaljer i allround-VM og enkeltdistanse-VM.\nTirsdag ble begge sendt til sykehus etter å ha vært innblandet i en sykkelvelt under trening i tyske Inzell, skriver Norges Skøyteforbund i en pressemelding.\nDe to skøyteløperne var på en treningstur da begge veltet i høy hastighet.\n– Skjedde fort\nSportssjef Lasse Sætre sier uhellet skjedde i forbindelse med noen sprintdrag.\n– Vi er litt usikre på hva som har skjedd, for det skjedde fort, sier Sætre til BT.\nDet opplyses at begge to er kraftig forslått. Det skal ha gått verst med Spieler Nilsen. Han ble sendt med luftambulanse til sykehuset i Traunstein. Der har han vært gjennom flere tester og undersøkelser. Det skal være snakk om en lettere hjernerystelse, i tillegg til en smell i ryggen i skrubbsår.\nSkøyteforbundet opplyser at han sannsynligvis skrives ut fra sykehuset enten torsdag eller fredag.\n– Dette var triste nyheter. Skader er dessverre en del av idretten, men dette er utrolig kjedelig for Simen, som også fikk ødelagt fjorårssesongen på grunn av skader, sier Sætre i pressemeldingen.\nUndersøker ryggen\nPedersen skal ha flere store skrubbsår, men ellers ingen påviste brudd eller alvorlige skader. Han må imidlertid holde seg i rolig trening noen uker. Han ble sendt til et lokalsykehus og skrives ut onsdag ettermiddag.\n– Vi prøver å få Sverre hjem i dag, men Simen må nok ligge der til vi finner ut hva som er problemet med ryggen, sier Sætre.\nLandslagssjef Bjarne Rykkje er på plass i Inzell. Han sier følgende til oss:\n– Det går fint, det er det jeg kan si. Alt kommer til å bli helt bra igjen.\n– Vil det påvirke oppkjøringen til sesongen?\n– Det er mulig. Det vet vi ikke helt enda. Det må vi finne ut.\nSamlingen skulle avsluttes søndag. Verdenscupen starter i Minsk i midten av november.
        QAS:
        1. Spørsmål: Hvor er landslagssjef Bjarne Rykkje?
           Svar: på plass i Inzell
        2. Spørsmål: Hvem ble sendt med luftambulanse til sykehuset?
           Svar: Spieler Nilsen
        3. Spørsmål: Når starter Verdenscupen i Minsk?
           Svar: i midten av november
        Teksten:
        Eleonore av Aquitaine\n:Ikke å forveksle med Leonora av Aquitaine\nEleonore, hertuginne av Aquitaine (født 1122, død 1. april 1204) var en av de rikeste og mektigste kvinnene i Vest-Europa i løpet av høymiddelalderen. Hun var beskytter av diktere og forfattere som Wace, Benoît de Sainte-Maure, og Chrétien de Troyes. Hun var gift med to konger, og ble selv mor til to konger.\nVåpenskjoldet til hertugdømmet Aquitaine.\nEleonore etterfulgte sin far som suo jure hertuginne av Aquitaine og grevinne av Poitiers da hun var femten år gammel, og ble med det den mest ettertraktede brud i Europa. Tre måneder etter sin tiltredelse giftet hun seg med Ludvig (VII), sønn og yngre medhersker av hennes verge, kong Ludvig VI av Frankrike. Som Frankrikes dronning deltok hun i det mislykkede andre korstog. Kort tid etter at korstoget var avsluttet ble Ludvig VII og Eleonore enige om å oppløse sitt ekteskap. Skilsmissen var dels motivert av hennes eget ønske, dels at de to barn hun hadde født begge var døtre, Marie og Alix, og ikke en etterlengtet sønn. Det kongelige ekteskap ble annullert den 11. mars 1152 med den oppgitte grunnen av blodsslektskap av fjerde grad. Deres døtre ble dog erklært legitime og varetekten ble gitt til deres far, mens Eleonores gods og riker som hun hadde tatt med seg inn i ekteskapet ble gitt tilbake til henne.\nSå snart hun kom tilbake til Poitiers fridde hun til den elleve år yngre Henrik, hertug av Normandie. Den 18. mai 1152, seks uker etter at hennes første ekteskap var blitt oppløst, giftet hun seg med Henrik. Den 25. oktober 1154 overtok hennes ektemann den engelske trone og ble konge av England, samt hersker over store områder i dagens Frankrike. I løpet av de neste tretten årene fødte hun Henrik II av England åtte barn: fem sønner, to som ble konger av England, og tre døtre. Imidlertid ble Henrik og Eleonore etter hvert politiske motstandere. Han lot henne fengsle i årene 1173 og 1189 fordi hun hadde støttet sønnens opprør mot ham.
        QAS:
        1. Spørsmål: Hvor mange konger var Eleonore, hertuginne av Aquitaine, gift med?
            Svar: to
        2. Spørsmål: Hvem var Eleonore, hertuginne av Aquitaine, gift med etter Ludvig VII?
            Svar: Henrik
        3. Spørsmål: Hva var grunnen til at Eleonore, hertuginne av Aquitaine, og Ludvig VII oppløste ekteskapet?
            Svar: blodsslektskap av fjerde grad
        """
        filename += '_few'
    else:
        prompt = 'Du må stille så mange forskjellige HV-spørsmål (hva, hvor, hvem, når, osv.) som mulig til teksten. Alle spørsmålene må ha et riktig svar i teksten som er kort og konsist, for eksempel en frase. Svaret må tas fra teksten akkurat som det står! Du kan IKKE omformulere svaret eller legge til dine egne ord. Svar på følgende format: Teksten: ...\nQAS:\n1. Spørsmål: ...\nSvar: ...\n2. Spørsmål: ...\nSvar: ...\n3. Spørsmål: ...\nSvar: ...'


    if args.prompt:
        filename += f'_prompt{args.prompt}'
    filename += '.jsonl'

    logging.info(f'Generation will be saved to {args.outpath}{filename}')
    processing_times = []
    for i, line in enumerate(load_dataset(args.datapath)):
        start_time = time.time()
        if i < int(args.run_from):  # Skip until we reach the start point
            continue
        try:
            if args.model == 'norallm/normistral-7b-warm-instruct':
                text = f"""<s><|im_start|> user
                    {prompt}\nTeksten:\n{line}QAS:<|im_end|>
                    <|im_start|> assistant
                    """
                out = zero_shot_instruct_normistral(text, tokenizer, model, device)
            elif args.model == 'Mistral-7B-Instruct-v0.2':
                text = f"<s>[INST]{prompt}\nTeksten:\n{line}[/INST]"
                out = zero_shot_instruct(text, tokenizer, model, device)
            elif args.model =='norallm/normistral-11b-warm':
                text = f"""
                    {prompt}\nTeksten:\n{line}\nQAS:\n
                    """
                out = zero_shot_normistral(text, tokenizer, model, device)
                logging.info(out)
            
            questions, answers = extract_qas(out)
            json_ready = format_qas(questions, answers, line, i+1)
            if len(questions) > 0:
                with open(args.outpath + filename, 'a', encoding='utf-8') as file:
                    file.write(json_ready + '\n')
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            torch.cuda.empty_cache()  # Releases unused memory
            
            if i % 100 == 0:
                logging.info(f'{i} instances processed.')
                # get average processing time for 100 instances
                avg_processing_time = (sum(processing_times)) / len(processing_times)
                processing_times = []
                logging.info(f"Avg. Processing time for article: {avg_processing_time:.4f} seconds")
        
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            continue    # Skip problematic entries and continue processing       
        if i >= int(args.run_from + args.num_instances):    # Stop after n lines
            break


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main()
