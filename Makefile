process:
	python -m src.Preprocessing

plots:.
	Rscript "src/Figures.R"