src := xlsr

check:
	black $(src) --check --diff
	isort $(src) --check --diff

format:
	black $(src)
	isort $(src)
