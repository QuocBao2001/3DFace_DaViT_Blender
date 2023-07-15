#from coarse_supervised_trainer import CSupTrain
from coarse_selfsup_trainer import CSelfTrain

#from fine_texture_trainer import FTextureTrain

from fine_texture_trainer_v2 import FTextureTrain

if __name__ == '__main__':
    # cSupTrain = CSupTrain()

    # cSupTrain.fit()

    # cSelfTrain = CSelfTrain()

    # cSelfTrain.fit()
    fTextureTrain = FTextureTrain()

    fTextureTrain.fit()