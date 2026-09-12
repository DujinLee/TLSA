from datasets.domain_net import DomainNet
from datasets.office31 import Office31
from datasets.officehome import OfficeHome
from datasets.visda import VisDa2017

dataset_classes = {
    "office31": Office31,
    "officehome": OfficeHome,
    "visda": VisDa2017,
    "domainnet": DomainNet,
}
