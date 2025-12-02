from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .MCW_KD import MCW_KD
from .MCW_KD_Dual import MCW_KD_Dual
from .DSKD import DualSpaceKDWithCMA
from .ULD import UniversalLogitDistillation
from .MinED import MinEditDisForwardKLD
from .MultiLevelOT import MultiLevelOT

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "MCW_KD": MCW_KD,
    "MCW_KD_Dual": MCW_KD_Dual,
    "MultiLevelOT": MultiLevelOT
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")