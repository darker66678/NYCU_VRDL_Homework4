if [ $1 == "train" ]; then
    python main_train_psnr.py --opt ./options/train_swinir_sr_classical.json 
elif [ $1 == "test" ]; then
    python main_test_swinir.py --task classical_sr --scale 3 --model_path $2 --folder_lq ../data/testing_lr_images/testing_lr_images/
fi