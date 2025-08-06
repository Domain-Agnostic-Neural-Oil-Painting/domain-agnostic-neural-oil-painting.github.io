CUDA_VISIBLE_DEVICES=1 python tta_train.py \
--name painter \
--gpu_ids 0 \
--model painter \
--dataset_mode null \
--batch_size  32 \
--lr 7.5e-3 \
--init_type normal \
--n_epochs 30 \
--n_epochs_decay 2 \
