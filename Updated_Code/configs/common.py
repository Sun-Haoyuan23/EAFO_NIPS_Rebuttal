output_root = "./experiments_300epoch"

epochs = 300
num_workers = 4
print_freq = 50
criterion = 'ce'
optimizer = 'adamw'
weight_decay = 5e-2
scheduler = 'cosine'
warmup_epochs = 20
warmup_lr = 1e-06
min_lr = 1e-05
evaluator = 'default'
save_interval = 300
clip_max_norm = 1.0
image_size = 224
amp = True
sync_bn = True
find_unused_params = False
need_targets = False