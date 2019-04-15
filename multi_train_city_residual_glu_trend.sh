date
for i in `seq 0 12`
do
{
python main.py --gpu $1 --model residual_city_glu --transform --t 30 --predict_days 1 --epochs 200 --lr 1e-4 --lr_schedule 30,150 --trend --test_time 10  --city_index $i
}&
done
wait
date
# python main.py --gpu $1 --model residual_city_glu --transform --t 30 --predict_days 1 --epochs 200 --lr 1e-4 --lr_schedule 30,150 --trend --test_time 10  --city_index 0
