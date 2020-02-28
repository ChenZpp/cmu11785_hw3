from phoneme_list import *
from ctcdecode import CTCBeamDecoder
import torch
import os

def CTCDecode(log_prob, input_length, phonemes = PHONEME_MAP, beamwidth = 10, num_processor = os.cpu_count()):

    phonemes = [" "] + phonemes
    decoder = CTCBeamDecoder(phonemes, beam_width=beamwidth, num_processes=int(num_processor), log_probs_input=True)

    log_prob = log_prob.transpose(0,1)
    input_length = input_length.type(torch.LongTensor)
    #print("The log_prob shape should be (batch size, largest length, num of classes)", log_prob.shape)
    #print("The log_prob should be a tensor: ", type(log_prob))
    #print("THe length should be a long tensor ", type(input_length))

    out, _, _, out_lens = decoder.decode(log_prob,input_length)
    result = []

    for i in range(out.shape[0]):
        res = ""
        for j in range(out_lens[i,0]):
            try:
                res += phonemes[int(out[i,0,j])]
            except:
                print(res)
                print(out[i,0,j])
        result.append(res)
    #print(result)
    return result
