from infer_coarse import CInfer
from utils.config import cfg

if __name__ == '__main__':
    cInfer = CInfer(texture_path=cfg.flame.dense_template_path)

    cInfer.fit()