import torch
from magvit2_pytorch import (
    VideoTokenizer,
    VideoTokenizerTrainer
)
import ipdb

tokenizer = VideoTokenizer(
    image_size = 64,  #change from 128
    init_dim = 64,
    max_dim = 512,
    codebook_size = 1024,
    layers = (
        'residual',
        'compress_space',
        ('consecutive_residual', 2),
        'compress_space',
        ('consecutive_residual', 2),
        'linear_attend_space',
        'compress_space',
        ('consecutive_residual', 2),
        'attend_space',
        'compress_time',
        ('consecutive_residual', 2),
        'compress_time',
        ('consecutive_residual', 2),
        'attend_time',
    )
    
    # discr_start_after_step = 1000.,
)

#importing weights from old training
#tokenizer.load_state_dict(torch.load("checkpoint.134.pt", map_location=torch.device('cpu')))


trainer = VideoTokenizerTrainer(
    tokenizer,
    use_wandb_tracking = True,
    dataset_folder = '../data/SegTrackv2_overfit/mp4s',     # folder of either videos or images, depending on setting below
    dataset_type = 'videos',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = 1,
    grad_accum_every = 8, #changed from 8
    learning_rate = 5e-3, #changed from 2e-4
    num_train_steps = 1_000_000,
    discr_start_after_step = 100000000.,
    checkpoint_every_step = 5000,
    validate_every_step = 100,
    num_frames = 9
)
# ipdb.set_trace()
torch.cuda.empty_cache()
# trainer.train()

with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
    trainer.train()

# after a lot of training ...
# can use the EMA of the tokenizer

ema_tokenizer = trainer.ema_tokenizer

# mock video

video = torch.randn(1, 3, 17, 128, 128)

# tokenizing video to discrete codes

codes = ema_tokenizer.tokenize(video) # (1, 9, 16, 16) <- in this example, time downsampled by 4x and space downsampled by 8x. flatten token ids for (non)-autoregressive training

# sanity check

decoded_video = ema_tokenizer.decode_from_code_indices(codes)

assert torch.allclose(
    decoded_video,
    ema_tokenizer(video, return_recon = True)
)
