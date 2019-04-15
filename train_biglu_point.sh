python main.py --gpu $1 --model biglu_point --transform --t 21 \
    --predict_days 1 --epochs 200 --lr 1e-3 --lr_schedule 30,150 \
        --test_time 10 --front --final  --trend --log_time 60 #71 #60 #21 \