#!/usr/bin/python
# Badri Adhikari, 5-21-2017
# Alex Morehead, 11-16-2020
# Train stage 1 models

from libcnntrain import *


dir_config = sys.argv[1]
file_weights = sys.argv[2]
string_header = sys.argv[3]
fileX = sys.argv[4]
fileX_stage2 = sys.argv[5]
L_MAX = int(sys.argv[7])
num_of_inputs_to_use = int(sys.argv[8])  # 200 maximum in initial training/validation batch

file_weights = dir_config + '/' + file_weights if file_weights else None
model_arch = read_model_arch(dir_config + '/model-arch.config')

print('')
print('SCRIPT        : ' + sys.argv[0])
print('dir_config    : ' + dir_config)
print('file_weights  : ' + file_weights if file_weights else "")
print('string_header : ' + string_header)
print('fileX         : ' + fileX)
print('fileX_stage2  : ' + fileX_stage2)
print('')

train_model(model_arch, file_weights, L_MAX, num_of_inputs_to_use)
