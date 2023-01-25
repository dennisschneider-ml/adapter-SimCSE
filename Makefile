#!/bin/bash
.ONESHELL: 

run:
	rm -rf ../temp-result-stack
	. ../.venv/bin/activate
	bash run_sup_example.sh --train_file data/nli_for_simcse.csv --model_name_or_path roberta-large --tokenizer_name roberta-large --pooler_type avg_first_last --per_device_train_batch_size 50 --learning_rate 1e-5 --output_dir ../temp-result-stack --eval_steps 250 --num_train_epochs 3

adapter:
	rm -rf ../temp-result-stack
	. ../.venv/bin/activate
	bash run_sup_example.sh --train_file data/nli_for_simcse.csv --model_name_or_path roberta-large --tokenizer_name roberta-large --pooler_type avg_first_last --per_device_train_batch_size 50 --output_dir ../temp-result-stack --eval_steps 250 --num_train_epochs 3 --adapter_model sts-b --adapter_config houlsby --learning_rate 1e-5 --freeze_adapter

continue:
	. ../.venv/bin/activate
	bash run_sup_example.sh --train_file data/nli_for_simcse.csv --model_name_or_path ../temp-result-stack --tokenizer_name roberta-large --pooler_type avg_first_last --per_device_train_batch_size 40 --output_dir ../end-pipeline-result-stack --eval_steps 1000 --max_steps 10000
	python evaluation.py --model_name_or_path ../end-pipeline-result-stack --pooler avg_first_last --task_set sts

evaluate:
	. ../.venv/bin/activate
	python evaluation.py --model_name_or_path ../temp-result-stack --pooler avg_first_last --task_set sts

install:
	python -m venv .venv
	pip install -r requirements.txt
	# Download training datasets.
	# For sup SimCSE.
	sh data/download_nli.sh
	# For unsup SimCSE.
	sh data/download_wiki.sh
	# Download evaluation datasets.
	sh SentEval/data/downstream/download_dataset.sh

