accumulative_counts = 8
batch_size = 2
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            pretrained_model_name_or_path=
            'models/InternVL3-8B',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.DatasetInfoHook'),
    dict(
        output_dir='./work_dirs/internvl3_8b_lora_refcoco/',
        type='xtuner.engine.hooks.loss_plot_hook.LossPlotHook'),
]
data_path = 'data/refcoco_train.json'
data_root = 'data/'
dataloader_num_workers = 2
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=-1,
        save_last=True,
        save_optimizer=False,
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=1,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_folder = 'data/'
launcher = 'pytorch'
llava_dataset = dict(
    data_paths=
    'data/refcoco_train.json',
    image_folders='data/',
    max_length=6144,
    model_path=
    'models/InternVL3-8B',
    template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
    type='xtuner.dataset.InternVL_V1_5_Dataset')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 5e-05
max_epochs = 20
max_length = 6144
max_norm = 1
model = dict(
    freeze_llm=True,
    freeze_visual_encoder=True,
    llm_lora=dict(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        target_modules=None,
        task_type='CAUSAL_LM',
        type='peft.LoraConfig'),
    model_path=
    'models/InternVL3-8B',
    type='xtuner.model.InternVL_V1_5')
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=5e-05,
        type='torch.optim.AdamW',
        weight_decay=0.0),
    type='DeepSpeedOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=0.6,
        start_factor=1e-05,
        type='mmengine.optim.LinearLR'),
    dict(
        begin=0.15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
path = 'models/InternVL3-8B'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.internlm2_chat'
randomness = dict(deterministic=False, seed=42)
resume = False
runner_type = 'FlexibleRunner'
strategy = dict(
    config=dict(
        bf16=dict(enabled=True),
        fp16=dict(enabled=False, initial_scale_power=16),
        gradient_accumulation_steps='auto',
        gradient_clipping='auto',
        train_micro_batch_size_per_gpu='auto',
        zero_allow_untested_optimizer=True,
        zero_force_ds_cpu_optimizer=False,
        zero_optimization=dict(overlap_comm=True, stage=2)),
    exclude_frozen_parameters=True,
    gradient_accumulation_steps=8,
    gradient_clipping=1,
    sequence_parallel_size=1,
    train_micro_batch_size_per_gpu=2,
    type='xtuner.engine.DeepSpeedStrategy')
tokenizer = dict(
    pretrained_model_name_or_path=
    'models/InternVL3-8B',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(max_epochs=20, type='xtuner.engine.runner.TrainLoop')
train_dataloader = dict(
    batch_size=2,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        data_paths=
        'data/refcoco_train.json',
        image_folders='data/',
        max_length=6144,
        model_path=
        'models/InternVL3-8B',
        template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
        type='xtuner.dataset.InternVL_V1_5_Dataset'),
    num_workers=2,
    sampler=dict(
        length_property='modality_length',
        per_device_batch_size=16,
        type='xtuner.dataset.samplers.LengthGroupedSampler'))
visualizer = None
warmup_ratio = 0.03
weight_decay = 0.0
work_dir = './work_dirs/internvl3_8b_lora_refcoco/'
