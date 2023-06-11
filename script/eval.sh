GPU=0
SEED=71
# EVAL_MODE=separate

# Returns to main directory
cd ../

# Computes human correlations
python eval.py \
    --gpu $GPU \
    --model bert_metric \
    --checkpoint_dir_path ./output/${SEED}/bert_metric_mlr_order_pretrain \
    --checkpoint_file_name model_best_dual_mlr_loss.ckpt \
    --pretrained_model_name bert-base-uncased
