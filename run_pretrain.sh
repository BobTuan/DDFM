# pretrain base model
python ./src/main.py \
--method Pretrain --mode pretrain --model_ckpt_path ./pretrain_ckpts/pretrain/ \
--data_path ../criteo/data.txt \

# pretrain ddfm
python ./src/main.py \
--method DDFM --mode pretrain --pretrain_ddfm_model_ckpt_path ./pretrain_ckpts/ddfm/ \
--data_path ../criteo/data.txt \