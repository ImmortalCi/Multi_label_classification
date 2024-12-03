export CUDA_VISIBLE_DEVICES=5
MODEL_NAME_OR_PATH=/home/jzm/gov_classification/bert-base-chinese
OUTPUT_DIR=./output/multi_label_v1
RUN_NAME=multi_label_v1

# OUTPUT_DIR=./output/debug
# RUN_NAME=debug

export WANDB_API_KEY=ea2ecdf4a828acb51d4ccf063512edd1f2f389d7

export WANDB_PROJECT=bert_CN-multilabel_classification
# export WANDB_PROJECT=debug

python main.py \
    --do_train true \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --class_file data/class_file.json \
    --all_file data/dataset.json \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 256 \
    --learning_rate 2e-05 \
    --warmup_steps 100 \
    --weight_decay 0. \
    --num_train_epochs 6 \
    --lr_scheduler_type "cosine" \
    --run_name $RUN_NAME \
    --overwrite_output_dir \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --seed 1 \
