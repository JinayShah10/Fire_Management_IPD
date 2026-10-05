.PHONY: setup pipeline fetch filter prescreen build report train evaluate ui api

setup:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

pipeline: fetch filter prescreen build report

fetch:
	PYTHONPATH=. ./venv/bin/python3 stage_010_data_ingestion/fetch_firms.py

filter:
	PYTHONPATH=. ./venv/bin/python3 stage_015_urban_filtering/urban_filter.py --download-workers 20 --workers 10

prescreen:
	PYTHONPATH=. ./venv/bin/python3 stage_020_prescreening/prescreen.py

build:
	PYTHONPATH=. ./venv/bin/python3 stage_030_dataset_builder/build_dataset.py

report:
	PYTHONPATH=. ./venv/bin/python3 tools/dataset_report.py
	PYTHONPATH=. ./venv/bin/python3 tools/check_scaling.py

train:
	PYTHONPATH=. ./venv/bin/python3 stage_040_model_training/train.py

evaluate:
	PYTHONPATH=. ./venv/bin/python3 stage_050_evaluation/evaluate.py

api:
	PYTHONPATH=. ./venv/bin/uvicorn api.main:app --reload --port 8000

ui:
	cd frontend && npm run dev
