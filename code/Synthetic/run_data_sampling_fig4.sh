export BIAS_MODEL_DEVICE=cpu


python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py 0 0 10.0 180 10000 FOURIER_101 FOURIER_201 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py 0 0 10.0 180 10000 FOURIER_102 FOURIER_202 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py 0 0 10.0 180 10000 FOURIER_103 FOURIER_203 2345



python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 1 0 10.0 180 10000 FOURIER_111 FOURIER_211 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 1 0 10.0 180 10000 FOURIER_112 FOURIER_212 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 1 0 10.0 180 10000 FOURIER_113 FOURIER_213 2345

python3 CounterfactualModel_VIZ.py 1 0 10.0 180 FOURIER_111 FOURIER_211
python3 CounterfactualModel_VIZ.py 1 0 10.0 180 FOURIER_112 FOURIER_212
python3 CounterfactualModel_VIZ.py 1 0 10.0 180 FOURIER_113 FOURIER_213

python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 2 0 10.0 180 10000 FOURIER_121 FOURIER_221 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 2 0 10.0 180 10000 FOURIER_122 FOURIER_222 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 2 0 10.0 180 10000 FOURIER_123 FOURIER_223 2345

python3 CounterfactualModel_VIZ.py 2 0 10.0 180 FOURIER_121 FOURIER_221
python3 CounterfactualModel_VIZ.py 2 0 10.0 180 FOURIER_122 FOURIER_222
python3 CounterfactualModel_VIZ.py 2 0 10.0 180 FOURIER_123 FOURIER_223

python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 4 0 10.0 180 10000 FOURIER_141 FOURIER_241 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 4 0 10.0 180 10000 FOURIER_142 FOURIER_242 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 4 0 10.0 180 10000 FOURIER_143 FOURIER_243 2345

python3 CounterfactualModel_VIZ.py 4 0 10.0 180 FOURIER_141 FOURIER_241
python3 CounterfactualModel_VIZ.py 4 0 10.0 180 FOURIER_142 FOURIER_242
python3 CounterfactualModel_VIZ.py 4 0 10.0 180 FOURIER_143 FOURIER_243

python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 6 0 10.0 180 10000 FOURIER_161 FOURIER_261 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 6 0 10.0 180 10000 FOURIER_162 FOURIER_262 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 6 0 10.0 180 10000 FOURIER_163 FOURIER_263 2345

python3 CounterfactualModel_VIZ.py 6 0 10.0 180 FOURIER_161 FOURIER_261
python3 CounterfactualModel_VIZ.py 6 0 10.0 180 FOURIER_162 FOURIER_262
python3 CounterfactualModel_VIZ.py 6 0 10.0 180 FOURIER_163 FOURIER_263

python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 8 0 10.0 180 10000 FOURIER_181 FOURIER_281 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 8 0 10.0 180 10000 FOURIER_182 FOURIER_282 2345
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 8 0 10.0 180 10000 FOURIER_183 FOURIER_283 2345

python3 CounterfactualModel_VIZ.py 8 0 10.0 180 FOURIER_181 FOURIER_281
python3 CounterfactualModel_VIZ.py 8 0 10.0 180 FOURIER_182 FOURIER_282
python3 CounterfactualModel_VIZ.py 8 0 10.0 180 FOURIER_183 FOURIER_283
