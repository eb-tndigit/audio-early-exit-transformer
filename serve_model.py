import os
import sys
import re
import string
import random
import json 
import io
from multipart import parse_form_data, is_form_request
from wsgiref.simple_server import make_server
from data_dh import get_infer_data_loader
from models.model.early_exit import Early_conformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.model_utils import *
from util.tokenizer import *
import digitalhub as dh


context_dict = {}

data_path="/data"

def evaluate_batch_ctc(args, model, batch, valid_len, inf):
    encoder = model(batch[0].to(args.device), valid_len)
    #print(f"encoder:{encoder}")
    best_combined = inf.ctc_cuda_predict(encoder[len(encoder)-1], args.tokens)
    #for CPU
    #best_combined = inf.ctc_predict_(encoder[len(encoder)-1], len(encoder)-1)
    print(f"best_combined:{best_combined}")
    if args.bpe == True:
        #transcript = apply_lex(args.sp.decode(best_combined[len(best_combined)-1][0].tokens).lower(), vocab)
        transcript = args.sp.decode(best_combined[len(best_combined)-1][0].tokens).lower()
    else:
        #transcript = apply_lex(re.sub(r"[#^$]+", "", best_combined[len(best_combined)-1].lower()), vocab)
        transcript = re.sub(r"[#^$]+", "", best_combined[len(best_combined)-1].lower())
    return transcript


def run(args, model, data_loader, inf):
    result = []
    for batch in data_loader:
        valid_len = batch[1]
        result.append(evaluate_batch_ctc(args, model, batch,
                            valid_len,  inf))

    return result


#def init(context, model_name="early-exit-model", sp_model="bpe-256.model", sp_lexicon="bpe-256.lex", sp_tokens="bpe-256.tok"):
def init(context, model_name="english-EE-conformer", sp_model="bpe-256.model", sp_lexicon="bpe-256.lex", sp_tokens="bpe-256.tok"):
    try:
        os.mkdir(data_path + "/upload")
        context.logger.info("create dir data/upload")
    except OSError as error:
        context.logger.warn(f"create dir data/upload error:{error}")
    try:
        os.mkdir(data_path + "/trained_model")
        context.logger.info("create dir data/trained_model")
    except OSError as error:
        context.logger.warn(f"create dir data/trained_model error:{error}")
    try:
        os.mkdir(data_path + "/sentencepiece")
        context.logger.info("create dir data/sentencepiece")
    except OSError as error:
        context.logger.warn(f"create dir data/sentencepiece error:{error}")

    #project = dh.get_or_create_project(os.getenv("PROJECT_NAME"))
    project = context.project

    sp_model_artifact = project.get_artifact(sp_model)    
    sp_model_path = sp_model_artifact.download(destination=data_path + "/sentencepiece", overwrite=True)

    sp_lexicon_artifact = project.get_artifact(sp_lexicon)    
    sp_lexicon_path = sp_lexicon_artifact.download(destination=data_path + "/sentencepiece", overwrite=True)

    sp_tokens_artifact = project.get_artifact(sp_tokens)    
    sp_tokens_path = sp_tokens_artifact.download(destination=data_path + "/sentencepiece", overwrite=True)

    args = get_args(initial_args=[], sp_model=sp_model_path, sp_lexicon=sp_lexicon_path, sp_tokens=sp_tokens_path)
    args.batch_size = 1
    args.n_workers = 1
    args.shuffle = False
    args.decoder_mode = 'ctc'
    args.model_type == 'early_conformer'

    context_dict['args'] = args

    # View the content of the files in the data path
    context.logger.info(f"Scanning contents of directory: {data_path}")

    try:
        from pathlib import Path
        # Define the root path to inspect
        root_path = Path(data_path)
        
        # Check if the path exists and is a directory
        if root_path.is_dir():
            context.logger.info(f"--- Contents of {data_path} ---")
            
            # Use rglob('*') to recursively find all files in all subdirectories
            all_files = sorted(list(root_path.rglob('*')))
            
            if not all_files:
                context.logger.info("Directory is empty.")
            
            for file_path in all_files:
                if file_path.is_file():
                    try:
                        # Get file size in bytes
                        size_in_bytes = file_path.stat().st_size
                        
                        # Convert bytes to megabytes (MB)
                        size_in_mb = size_in_bytes / (1024 * 1024)
                        
                        # Get the path relative to the data_path for cleaner output
                        relative_path = str(file_path.relative_to(root_path))
                        
                        # Log the formatted output
                        log_message = f"{relative_path:<60} {size_in_mb:>8.3f} MB"
                        context.logger.info(log_message)
                    
                    except FileNotFoundError:
                        # This can happen in rare cases with broken symbolic links
                        context.logger.warn(f"{file_path.relative_to(root_path):<60} File not found or broken link.")
        else:
            context.logger.error(f"Error: The specified data_path '{data_path}' does not exist or is not a directory.")
            # Solleva un'eccezione per far fallire l'inizializzazione se il percorso è critico
            raise FileNotFoundError(f"Required directory not found: {data_path}")
            
    except Exception as e:
        context.logger.error(f"An unexpected error occurred during file scanning: {str(e)}")
        raise # Rilancia l'eccezione per vedere l'errore completo nei log

    context.logger.info("--- End of content view ---")
    
    # --- 3. Assicurati che qui ci sia il resto del tuo codice di inizializzazione ---
    # Esempio:
    # context.user_data.model = load_my_model()

    context.logger.info("Context initialization complete.")

    model = project.get_model(model_name)
    path = model.download(destination=data_path + "/trained_model", overwrite=True)
    model = load_model(path, args)
    context_dict['model'] = model

    
    # load lexicon
    #lexicon_artifact = project.get_artifact(lexicon)    
    #lexicon_path = lexicon_artifact.download(destination=data_path + "/sentencepiece", overwrite=True)
    #vocab = load_dict(lexicon_path)
    #context_dict['vocab'] = vocab

    context.logger.info(f"init:{len(context_dict)}")
    
    setattr(context, "context_dict", context_dict)


def load_model(model_path, args):
    model = Early_conformer(src_pad_idx=args.src_pad_idx,
                            n_enc_exits=args.n_enc_exits,
                            d_model=args.d_model,
                            enc_voc_size=args.enc_voc_size,
                            dec_voc_size=args.dec_voc_size,
                            max_len=args.max_len,
                            d_feed_forward=args.d_feed_forward,
                            n_head=args.n_heads,
                            n_enc_layers=args.n_enc_layers_per_exit,
                            features_length=args.n_mels,
                            drop_prob=args.drop_prob,
                            depthwise_kernel_size=args.depthwise_kernel_size,
                            device=args.device).to(args.device)
    
    model.load_state_dict(torch.load(model_path, map_location=args.device))

    model.eval()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print("batch_size:", batch_size, " num_heads:", n_heads, " num_encoder_layers:",
    #     n_enc_layers, "vocab_size:", dec_voc_size, "DEVICE:", device)

    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)

    return model


def serve_local(context_dict, path):
    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=context_dict['args'])

    paths = []
    paths.append(path)
    data_loader = get_infer_data_loader(args=context_dict['args'], paths=paths)

    result = run(model=context_dict['model'], args=context_dict['args'], data_loader=data_loader,
        inf=inf)
    
    for num, transcript in enumerate(result):
        print(f"Transcript[{num}]:{transcript}")

    if len(result) >= 1: 
        return result[0]
    else:
        return ""


def serve(context, event):
    context.logger.info(f"Received event: {event.body}")
    artifact_name = event.body["name"]
    artifact = context.project.get_artifact(artifact_name)    
    path = artifact.download(destination=data_path + "/upload", overwrite=True)
    
    transcript = serve_local(context.context_dict, path)
    context.logger.warn(f"Transcript for file {path}:{transcript}")

    results = []
    info = {}
    info['filename'] = artifact_name
    info['transcript'] = transcript
    results.append(info)

    return results
       

def serve_multipart(context, event):
    try:
        content_type = event.headers.get('Content-Type', '')
        context.logger.info(f"Received multipart event: {content_type}")
        results = []
        #environ = io.BytesIO(event.body)
        #environ = event.body
        if 'multipart/form-data' in content_type:
            context.logger.info("serve multipart buffer")
            environ = {
                "wsgi.input": io.BytesIO(event.body),
                "CONTENT_LENGTH": str(len(event.body)),
                "CONTENT_TYPE": content_type,
                "REQUEST_METHOD": "POST",
            }            
            forms, files = parse_form_data(environ)
            context.logger.info("serve multipart files")
            for filed_name in files:
                file_details = files[filed_name]
                context.logger.info(f"process file:{file_details.filename}")
                filename = data_path + "/upload/" + id_generator() + "_" + file_details.filename
                file_details.save_as(filename)
                context.logger.info(f"filename:{filename}") 

                transcript = serve_local(context.context_dict, filename)  
                info = {}
                info['filename'] = filed_name
                info['transcript'] = transcript
                results.append(info)

                if os.path.exists(filename):
                    os.remove(filename)
        
        return results
    except Exception as e:
        context.logger.error(f"serve_multipart error:{e}")
        return context.Response(body=f"Error:{e}", status_code=500)



def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def simple_app(environ, start_response):
    results = []
    if is_form_request(environ):
        forms, files = parse_form_data(environ)
        for filed_name in files:
            try:
                file_details = files[filed_name]
                print(f"process file:{file_details.filename}")
                filename = data_path + "/upload/" + id_generator() + "_" + file_details.filename
                file_details.save_as(context_dict, filename) 

                transcript = serve_local(filename)  
                info = {}
                info['filename'] = filename
                info['transcript'] = transcript
                results.append(info)

                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print(e)

    status = '200 OK'
    headers = [('Content-type', 'application/json; charset=utf-8')]
    content = json.dumps(results)
    content = [content.encode('utf-8')]
    start_response(status, headers)
    return content


def main():
    args = sys.argv[1:] 
    if len(args) > 0:
        init(model_name=args[0])
    else:
        init()    

    with make_server('', 8051, simple_app) as httpd:
        print("Serving on port 8051...")
        httpd.serve_forever()


if __name__ == "__main__":
    main()