# import packages
import os
from cec17_functions import cec17_test_func
import numpy as np
from copy import deepcopy


DimSize = 100
PopSizeMax = 18 * DimSize
PopSizeMin = 6
PopSize = PopSizeMax
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1
phi = 0.15

FitBest = 0
BestIndi = None

def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


# initialize the M randomly
def Initialization():
    global Pop, Velocity, FitPop, PopSize, PopSizeMax, FitBest, BestIndi
    PopSize = PopSizeMax
    Pop = np.zeros((PopSize, DimSize))
    FitPop = np.zeros(PopSize)
    Velocity = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])
    best_idx = np.argmin(FitPop)
    FitBest = FitPop[best_idx]
    BestIndi = deepcopy(Pop[best_idx])


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def LICSO():
    global Pop, Velocity, FitPop, phi, PopSize, curFEs, FitBest, BestIndi

    sequence = list(range(PopSize))
    np.random.shuffle(sequence)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    Xmean = np.mean(Pop, axis=0)
    for i in range(int(PopSize / 3)):
        idx1 = sequence[3 * i]
        idx2 = sequence[3 * i + 1]
        idx3 = sequence[3 * i + 2]

        idxes = [idx1, idx2, idx3]
        sort_idx = np.argsort([FitPop[idx1], FitPop[idx2], FitPop[idx3]])
        best_idx = idxes[sort_idx[0]]
        sbest_idx = idxes[sort_idx[1]]
        worst_idx = idxes[sort_idx[2]]

        Off[best_idx] = deepcopy(Pop[best_idx])  # Best: Copy
        FitOff[best_idx] = FitPop[best_idx]

        if np.random.rand() < 0.5:   # 2-Best: Copy or Move
            Velocity[sbest_idx] = np.random.rand(DimSize) * Velocity[sbest_idx] + np.random.rand(DimSize) * (
                    Pop[best_idx] - Pop[sbest_idx]) + phi * (Xmean - Pop[sbest_idx])
            Off[sbest_idx] = Pop[sbest_idx] + Velocity[sbest_idx]
            Off[sbest_idx] = Check(Off[sbest_idx])
            FitOff[sbest_idx] = fitness(Off[sbest_idx])
            if FitOff[sbest_idx] < FitBest:
                FitBest = FitOff[sbest_idx]
                BestIndi = deepcopy(Off[sbest_idx])
            curFEs += 1
        else:
            Off[sbest_idx] = deepcopy(Pop[sbest_idx])
            FitOff[sbest_idx] = FitPop[sbest_idx]

        Velocity[worst_idx] = np.random.rand(DimSize) * Velocity[worst_idx] + np.random.rand(DimSize) * (
                Pop[best_idx] - Pop[worst_idx]) + phi * (BestIndi - Pop[worst_idx])
        Off[worst_idx] = Pop[worst_idx] + Velocity[worst_idx]
        Off[worst_idx] = Check(Off[worst_idx])
        FitOff[worst_idx] = fitness(Off[worst_idx])
        if FitOff[worst_idx] < FitBest:
            FitBest = FitOff[worst_idx]
            BestIndi = deepcopy(Off[worst_idx])
        curFEs += 1

    PopSize = round(((PopSizeMin - PopSizeMax) / MaxFEs * curFEs + PopSizeMax))
    if PopSize % 3 == 1:
        PopSize += 2
    elif PopSize % 3 == 2:
        PopSize += 1

    PopSize = max(PopSize, PopSizeMin)

    sorted_idx = np.argsort(FitOff)
    Pop = deepcopy(Off[sorted_idx[0:PopSize]])
    FitPop = FitOff[sorted_idx[0:PopSize]]


def RunLICSO():
    global FitPop, curFEs, TrialRuns, DimSize
    All_Trial = []
    MAX = 0
    for i in range(TrialRuns):
        BestList = []
        curFEs = 0
        np.random.seed(88 + 20 * i)
        Initialization()
        BestList.append(min(FitPop))
        while curFEs < MaxFEs:
            LICSO()
            BestList.append(min(FitPop))
        MAX = max(MAX, len(BestList))
        All_Trial.append(BestList)

    for i in range(len(All_Trial)):
        for j in range(len(All_Trial[i]), MAX):
            All_Trial[i].append(All_Trial[i][-1])

    np.savetxt("./LICSO_Data/CEC2017/F" + str(FuncNum) + "_" + str(Dim) + "D.csv", All_Trial, delimiter=",")



def main(dim):
    global FuncNum, DimSize, MaxFEs, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    LB = [-100] * dim
    UB = [100] * dim

    for i in range(1, 31):
        if i == 2:
            continue
        FuncNum = i
        RunLICSO()


if __name__ == "__main__":
    if os.path.exists('./LICSO_Data/CEC2017') == False:
        os.makedirs('./LICSO_Data/CEC2017')
    Dims = [100]
    for Dim in Dims:
        main(Dim)


