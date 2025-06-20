## Serving a model
In order to start the inference service, you should upload to the platform the required files.
The files could be found on Huggingface
- for English: https://huggingface.co/SpeechTek/English-EE-conformer/tree/main
- for Italian: https://huggingface.co/SpeechTek/Italian-EE-conformer/tree/main

0. Git clone from English repo
```Bash
git clone https://huggingface.co/SpeechTek/English-EE-conformer/tree/main
```


1. Initializing the project
 ```Python
import digitalhub as dh

project = dh.get_or_create_project("demo-early-exit-eng")
```


2. Store the required files as project artifacts
 ```Python
 project.log_model(
    name="early-exit-model",
    kind="model",
    source="./English-EE-conformer/english-EE-conformer",
    algorithm="early-exit",
    framework="pytorch"
)

project.log_artifact(
    name="bpe-256.model",
    kind="artifact",
    source="./English-EE-conformer/bpe-256.model"
)

project.log_artifact(
    name="bpe-256.lex",
    kind="artifact",
    source="./English-EE-conformer/bpe-256.lex"
)

project.log_artifact(
    name="bpe-256.tok",
    kind="artifact",
    source="./English-EE-conformer/bpe-256.tok"
)
```


3. Define a function that handle the serving process
```Python
func = project.new_function(name="serve_function",
                            kind="python",
                            python_version="PYTHON3_10",
                            code_src="git+https://github.com/tn-aixpa/audio-early-exit-transformer",
                            handler="serve_model:serve_multipart",
                            init_function="init",
                            requirements=["torch==2.5.0", "torchaudio==2.5.0", "tensorboard==2.18.0",
                                "flashlight==0.1.1", "flashlight-text==0.0.7", "sentencepiece==0.2.0",
                                "soundfile==0.12.1", "editdistance==0.8.1", "multipart==1.2.1"])
```


4. Run the serving function
```Python
run = func.run(
    action="serve",
    profile="1xa100",
    resources = {"mem":{"requests": "2Gi",}},
    init_parameters = {
        "model_name":"early-exit-model",
        "sp_model":"bpe-256.model",
        "sp_lexicon":"bpe-256.lex",
        "sp_tokens":"bpe-256.tok"
    },    
    volumes=[
        {
            "volume_type": "persistent_volume_claim",
            "name": "early-exit-demo-serve",
            "mount_path": "/data",
            "spec": { "size": "1Gi" }        
        }
    ]
)
```
As parameters you must set the name of artifacts that you have previousply uploaded.


5. In order to invoke the service, an HTTP form-multipart post request must be invoked
```Python
import requests

file_path = "test-audio.wav"

run.refresh()
url = "http://" + run.status.service['url']

with open(file_path, "rb") as file:
    files = {'file': file}
    response =  run.invoke(files=files, method='POST')

print(f"response code:{response.status_code}")
print(f"response body:{response.text}")
```

