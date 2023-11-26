# DEMD ++Multi-marginal Wasserstein GAN

This code extends the original code in "Multi-marginal Wasserstein GAN" to DEMD.

```
    pipenv install
    pipenv shell
```

The below instructions are taken from the original MWGAN repository [MWGAN](https://github.com/caojiezhang/MWGAN).

### Data preparation

1. Download the CelebA dataset and corresponding attibute labels.
    * Link: [Dropbox](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0) or [BaiduNetdisk](https://pan.baidu.com/s/18_IHVDbA1PW5ljN_Fg84YA&shfl=sharepset)      
    * Put the data in `./data/celeba` directory

2. Construct the facial attribute translation dataset (i.e. Black_Hair, Blond_Hair, Eyeglasses, Mustache, Pale_Skin).
        
        python data_process.py --process celeba --source_attr Black_Hair

3. Construct the edge -> celeba dataset (i.e. Edge, Black_Hair, Blond_Hair, Brown_Hair).
    * Organize data using folder structure described [here](material/data_structure.md).
        * Get  Black_Hair, Blond_Hair data from **step 2**.

        * Get Brown_Hair data:

                python data_process.py --process celeba \
                    --selected_attrs Brown_Hair --target_dir data/Edge2Celeba

        * Get Edge data:

                python data_process.py --process edge \
                    --source_dir data/Edge2Celeba --target_dir data/Edge2Celeba\
                    --selected_attrs Black_Hair Blond_Hair Brown_Hair --select_nums 15000


### Training

To train MWGAN on facial attribute translation task:

    python main.py --num_domains 5 --batch_size 16 \
        --data_root data/Celeba5domain/train --src_domain Black_Hair \
        --result_root results_celeba \
        --lambda_cls 1 --lambda_info 20 --lambda_idt 10
To train MWGAN on edge->celeba task:

    python main.py --num_domains 4 --batch_size 16 \
        --data_root data/Edge2Celeba/train --src_domain Edge \
        --result_root results_edge \
        --lambda_cls 10 --lambda_info 10 --cls_loss BCE

* if you don't have tensorboardX and tensorflow, please add `--use_tensorboard false`

### Testing

To test MWGAN on facial attribute translation task:

    python main.py --mode test --num_domains 5 --batch_size 16 \
        --data_root data/Celeba5domain/test --src_domain Black_Hair \
        --result_root results_celeba

To test MWGAN on edge->celeba task:

    python main.py --mode test --num_domains 4 --batch_size 16 \
        --data_root data/Edge2Celeba/test --src_domain Edge \
        --result_root results_edge

## DEMD Runs

```
python main.py --num_domains 5 --batch_size 16 --data_root data/Celeba5domain/train --src_domain Black_Hair --result_root mwgan_results  --lambda_cls 1 --lambda_info 20 --lambda_idt 10 --use_tensorboard true --regType 3M --lambda_reg 100 --demd_reg 0 --demd_nbins 10 --seed 1234
```

```
python main.py --num_domains 5 --batch_size 16 --data_root data/Celeba5domain/train --src_domain Black_Hair --result_root demd_results  --lambda_cls 1 --lambda_info 20 --lambda_idt 10 --use_tensorboard true --regType demd --lambda_reg 0 --demd_reg 1000 --demd_nbins 10 --seed 1234
```