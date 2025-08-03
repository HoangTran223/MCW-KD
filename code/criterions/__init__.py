from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .MCW_KD import MCW_KD
from .DSKD import DualSpaceKDWithCMA
from .ULD import UniversalLogitDistillation
from .MinED import MinEditDisForwardKLD
from .dual_space_kd_new_kb1 import DualSpaceKDWithCMA_OT_1
from .dual_space_kd_new_kb2 import DualSpaceKDWithCMA_OT_2
from .ULD_1 import UniversalLogitDistillation_1
from .MinED_1 import MinEditDisForwardKLD_1
from .MultiLevelOT import MultiLevelOT
from .MultiLevelOT_1 import MultiLevelOT_1

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "MCW_KD": MCW_KD,
    "MultiLevelOT": MultiLevelOT
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")