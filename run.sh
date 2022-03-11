export TRAIN=/h/u1/cs401/A2/data/Hansard/Training/
export TEST=/h/u1/cs401/A2/data/Hansard/Testing/
# 1. Generate vocabularies
python3.9 a2_run.py vocab $TRAIN e vocab.e.gz
python3.9 a2_run.py vocab $TRAIN f vocab.f.gz
# 2. Split train and dev sets
python3.9 a2_run.py split $TRAIN train.txt.gz dev.txt.gz
# 3. Train a model without attention
srun -p csc401 --gres gpu \
python3.9 a2_run.py train $TRAIN \
vocab.e.gz vocab.f.gz \
train.txt.gz dev.txt.gz \
model_wo_att.pt.gz \
--device cuda
# 6. Test the model without attention
srun -p csc401 --gres gpu \
python3.9 a2_run.py test $TEST \
vocab.e.gz vocab.f.gz model_wo_att.pt.gz \
--device cuda
# 4. Train a model with attention
srun -p csc401 --gres gpu \
python3.9 a2_run.py train $TRAIN \
vocab.e.gz vocab.f.gz \
train.txt.gz dev.txt.gz \
model_w_att.pt.gz \
--with-attention \
--device cuda
# 7. Test the model with attention
srun -p csc401 --gres gpu \
python3.9 a2_run.py test $TEST \
vocab.e.gz vocab.f.gz model_w_att.pt.gz \
--with-attention --device cuda
# 5. Train a model with multi-head attention
srun -p csc401 --gres gpu \
python3.9 a2_run.py train $TRAIN \
vocab.e.gz vocab.f.gz \
train.txt.gz dev.txt.gz \
model_w_mhatt.pt.gz \
--with-multihead-attention \
--device cuda
# 8. Test the model with multi-head attention
srun -p csc401 --gres gpu \
python3.9 a2_run.py test $TEST \
vocab.e.gz vocab.f.gz model_w_mhatt.pt.gz \
--with-multihead-attention --device cuda