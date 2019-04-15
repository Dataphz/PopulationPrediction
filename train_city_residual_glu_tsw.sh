python main.py --gpu $1 --model residual_city_glu \
        --transform --trend --t 30 --predict_days 1  \
        --epochs 200 --lr 1e-3 --lr_schedule 30,150 \
        --test_time 10 --front --final \
        --season --weighted --log_time 63 #--residual