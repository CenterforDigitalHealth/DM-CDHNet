# Non-head-only objective-guard pass from the selected single checkpoint.
# Uses the last block, final norm, and heads, while increasing DATA loss
# weight to preserve KR disease AUC during SHIFT/TOTAL/TIME optimization.

model_size = 'small'
n_layer = 8
n_head = 8
n_kv_head = 4
n_embd = 256
block_size = 512
batch_size = 128
gradient_accumulation_steps = 1

init_from = 'finetune'
finetune_ckpt_path = 'out/single_ckpt_0505/transplants/w80_teacherdrug_total_time_from_aux1000.pt'

learning_rate = 6.0e-6
max_iters = 250
warmup_iters = 50
lr_decay_iters = 250
min_lr = 8.0e-7
eval_interval = 50
eval_iters = 60
early_stop_patience_iters = 250
dropout = 0.20
weight_decay = 0.04

ignore_tokens = [0]
data_loss_static_weight = 0.10
data_loss_disease_weight = 1.0
data_loss_drug_weight = 0.20

trainable_parameter_patterns = [
    'h.7.*',
    'ln_f.*',
    'multi_head.*',
]

drug_token_only_shift = True
drug_token_only_total = True
shift_loss_type = 'focal'
num_shift_classes = 2
shift_focal_gamma = 2.0
separate_shift_na_from_padding = False

use_moe = True
num_experts = 8
experts_per_token = 2
moe_expert_type = 'gelu'
sliding_window = 128
use_drug_conditioning = True
use_teacher_forcing_drug_cond = True

loss_weight_data = 1.0
loss_weight_shift = 10.0
loss_weight_total = 10.0
loss_weight_time = 1.0

mdn_n_components = 16
mdn_log_s_min = -0.5

save_eval_checkpoints = True
save_eval_checkpoint_interval = 50
run_internal_test_after_training = False

out_dir = ''
out_dir_use_timestamp = False
