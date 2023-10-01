.PHONY: cifar10 flickr30k ffhq

DEBUG ?= True
RUN_ID ?=

OUTPUT := logs
DATA_DIR := data
CONFIG := configs

${OUTPUT}/cifar10_default/${RUN_ID}/${RUN_ID}.ckpt:
	python autoencoder.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config=${CONFIG}/cifar10_default.yaml


${OUTPUT}/flickr30k_default/${RUN_ID}/${RUN_ID}.ckpt:
	python autoencoder.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config=${CONFIG}/flickr30k_default.yaml

${OUTPUT}/ffhq_default/${RUN_ID}/${RUN_ID}.ckpt:
	python autoencoder.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config=${CONFIG}/ffhq_default.yaml

${OUTPUT}/ffhq_default/${RUN_ID}/image_tokens.pt: ${OUTPUT}/ffhq_default/${RUN_ID}/${RUN_ID}.ckpt
	python image_tokenizer.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config=${CONFIG}/ffhq_default.yaml

${OUTPUT}/ffhq_default/${RUN_ID}/generator_weights.ckpt: ${OUTPUT}/ffhq_default/${RUN_ID}/image_tokens.pt
	python generator.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config=${CONFIG}/ffhq_default.yaml 

cifar10_%: ${OUTPUT}/cifar10_%/${RUN_ID}/${RUN_ID}.ckpt
	:

cifar10: cifar10_default

flickr30k_%: ${OUTPUT}/flickr30k_%/${RUN_ID}/${RUN_ID}.ckpt
	:

flickr30k: flickr30k_default

ffhq_%: ${OUTPUT}/ffhq_%/${RUN_ID}/generator_weights.ckpt
	:

ffhq: ffhq_default