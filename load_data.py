# Code reused from https://github.com/arghosh/AKT.git
import numpy as np
import math


class DATA(object):
    def __init__(self, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.X = None
    

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    it sequence
    at sequence
    '''

    def load_data(self, path):
        f_data = open(path, 'r')
        k_data = []
        a_data = []
        e_data = []
        it_data = []
        at_data = []
        al_data = []
        df_data = []
        l_data = []
        

        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 8 != 0:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0:
                    line_data = line_data[:-1]


            if lineID % 8 ==1:
                K = line_data
            elif lineID % 8 == 2:
                A = line_data
            elif lineID % 8 == 3:
                E = line_data
            elif lineID % 8 == 4:
                IT = line_data
            elif lineID % 8 == 5:
                AT = line_data
            elif lineID % 8 == 6:
                AL = line_data
            elif lineID % 8 == 7:
                DF = line_data
            
                
                # start split the data
                n_split = 1
                total_len = len(A)
                if total_len > self.seqlen:
                    n_split = math.floor(len(A) / self.seqlen)
                    if total_len % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    kc_sequence = []
                    answer_sequence = []
                    exercise_sequence = []
                    it_sequence = []
                    at_sequence = []
                    al_sequence = []
                    df_sequence = []
                    
                    if k == n_split - 1:
                        end_index = total_len
                    else:
                        end_index = (k + 1) * self.seqlen
                    # choose the sequence length is larger than 2
                    if end_index - k * self.seqlen > 2:
                        for i in range(k * self.seqlen, end_index):
                            kc_sequence.append(int(K[i]))
                            answer_sequence.append(int(A[i]))
                            exercise_sequence.append(int(E[i]))
                            """ try:
                                it_sequence.append(int(IT[i]))
                            except:
                                print('qqq') """
                            it_sequence.append(int(IT[i]))
                            at_sequence.append(int(AT[i]))
                            al_sequence.append(int(AL[i]))
                            df_sequence.append(int(DF[i]))

                            

                        k_data.append(kc_sequence)
                        a_data.append(answer_sequence)
                        e_data.append(exercise_sequence)
                        it_data.append(it_sequence)
                        at_data.append(at_sequence)
                        al_data.append(al_sequence)
                        df_data.append(df_sequence)
                       
        f_data.close()

        k_dataArray = np.zeros((len(k_data), self.seqlen))
        for j in range(len(k_data)):
            dat = k_data[j]
            k_dataArray[j, :len(dat)] = dat

        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        it_dataArray = np.zeros((len(it_data), self.seqlen))
        for j in range(len(it_data)):
            dat = it_data[j]
            it_dataArray[j, :len(dat)] = dat

        at_dataArray = np.zeros((len(at_data), self.seqlen))
        for j in range(len(at_data)):
            dat = at_data[j]
            at_dataArray[j, :len(dat)] = dat

        al_dataArray = np.zeros((len(al_data), self.seqlen))
        for j in range(len(al_data)):
            dat = al_data[j]
            al_dataArray[j, :len(dat)] = dat

        df_dataArray = np.zeros((len(df_data), self.seqlen))
        for j in range(len(df_data)):
            dat = df_data[j]
            df_dataArray[j, :len(dat)] = dat
        
        

        return k_dataArray, a_dataArray, e_dataArray, it_dataArray, at_dataArray ,al_dataArray, df_dataArray