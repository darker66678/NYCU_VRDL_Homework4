from ISR.train import Trainer
from ISR.models import RRDN, RDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19

lr_train_patch_size = 25
layers_to_extract = [5, 9]
scale = 3
hr_train_patch_size = lr_train_patch_size * scale

rrdn = RDN(arch_params={'C': 4, 'D': 12, 'G': 64, 'G0': 64,
           'x': scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size,
                  layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

loss_weights = {
  'generator': 0.9,
  'feature_extractor': 0.0833,
  'discriminator': 0.01
}
losses = {
    'generator': 'mae',
    'feature_extractor': 'mse',
    'discriminator': 'binary_crossentropy'
}

log_dirs = {'logs': './logs', 'weights': './weights'}

learning_rate = {'initial_value': 0.001,
                 'decay_factor': 0.4, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

trainer = Trainer(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='../data/train/lr/',
    hr_train_dir='../data/train/hr/',
    lr_valid_dir='../data/val/lr/',
    hr_valid_dir='../data/val/hr/',
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    dataname='image_dataset',
    log_dirs=log_dirs,
    weights_generator=None,
    weights_discriminator=None,
    n_validation=5,
)

trainer.train(
    epochs=50,
    steps_per_epoch=500,
    batch_size=4,
    monitored_metrics={'val_generator_PSNR_Y': 'max'}
)
