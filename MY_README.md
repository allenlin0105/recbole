# Steps

1. Set up virtual environment
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

2. Log in to wandb (prepare your API key)
```bash
$ wandb login
```

3. Modify `config/Default.yaml`
```bash
wandb_entity: $wandb_acc_name
wandb_project: $wandb_project_name
```

4. Training (Config file is stored at `config/model/{model}.yaml`)
```bash
# Use model "Caser" and dataset "lastfm" as example
$ python run.py -m Caser -d lastfm
# Format
$ python run.py -m {model} -d {dataset}
```

5. Hyperparameter Tuning (Config file is stored at `config/hyper/{dataset}.hyper`)
```bash
# Use model "Caser" and dataset "lastfm" as example
$ python run_hyper.py --model=Caser --dataset=lastfm --gpu_id=0
# Format
$ python run_hyper.py --model={model} --dataset={dataset} --gpu_id={gpu_id}
```
