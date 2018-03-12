docker run  --rm \
            -it \
            --runtime=nvidia \
            $1 \
            python magnet_loss_test.py --lr 1e-4 --mnist --batch-size 64 --magnet-loss
