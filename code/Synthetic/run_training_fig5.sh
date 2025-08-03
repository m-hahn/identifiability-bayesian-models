export BIAS_MODEL_DEVICE=cuda

# To run the fitting command separately, comment out the unwanted command lines
# P=2,4,6,8
python3 runForFigure5.py

# P=1
python3 runForFigure5_L1.py

# P=0
python3 runForFigure5_Zero.py
