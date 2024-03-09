import os
import time
from typing import Optional

import torch
from prettytable import PrettyTable
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Logging(SummaryWriter):

    def __init__(self, path: str):
        super(Logging, self).__init__(path)
        self.txt_logger_path = os.path.join(path, 'txt_logger_output.txt')
        self.config_logger_path = os.path.join(path, 'config_logger_output.py')

    def save_model(self, model: nn.Module, rank: str = None) -> None:
        if rank is not None:
            save_path = os.path.join(self.log_dir, model.__class__.__name__ + rank + '.pth')
        else:
            save_path = os.path.join(self.log_dir, model.__class__.__name__ + '.pth')
        torch.save(model.state_dict(), save_path)

    def save_file(self, dic, epoch: Optional[int], prefix: str = None) -> None:
        def logging(s: str) -> None:
            print(s)
            with open(self.txt_logger_path, mode='a') as f:
                f.write('[' + time.asctime(time.localtime(time.time())) + ']    ' + s + '\n')
                f.close()

        def _round(num: object, length: int = 5) -> object:
            if isinstance(num, float):
                return round(num, length)
            else:
                return num

        # Log in the tensorboard.
        if isinstance(dic, str):
            logging(dic)
        elif isinstance(dic, dict):
            for k in dic.keys():
                self.add_scalar(f'{prefix}/{k}', dic[k], epoch)

            # Log in the command line.
            tupled_dict = [(k, v) for k, v in dic.items()]
            tb = PrettyTable(["Prefix", "Epoch"] + [ii[0] for ii in tupled_dict])
            tb.add_row([prefix, epoch] + [_round(ii[1]) for ii in tupled_dict])
            txt_info = str(tb)
            logging(txt_info)
        else:
            raise TypeError("the type of dic should be str or dict, but got {}".format(type(dic)))

    def save_config(self, config: dict) -> None:
        """
            Overview:
                Save configuration to python file
            Arguments:
                - config: Config dict
                - path: Path of saved file
            """
        config_string = str(config)
        from yapf.yapflib.yapf_api import FormatCode
        config_string, _ = FormatCode(config_string)
        config_string = config_string.replace('inf,', 'float("inf"),')
        with open(self.config_logger_path, "a") as f:
            f.write('exp_config = ' + config_string)
            f.close()
