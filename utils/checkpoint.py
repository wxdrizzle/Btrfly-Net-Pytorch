from utils.logger import *
import os
import torch

class CheckPointer:
    _last_checkpoint_name = 'last_checkpoint.txt'
    _best_checkpoint_name = 'best_checkpoint.txt'
    def __init__(self, model, optimizer, save_dir=""):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.logger = setup_colorful_logger("checkpointer", save_dir=os.path.join(save_dir, 'log.txt'), format="include_other_info")

    def save(self, name, is_last, is_best, **kwargs):
        if not self.save_dir:
            return

        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file, is_last=is_last, is_best=is_best)

    def load(self, f=None, use_latest=True, is_val=False):
        if self.has_checkpoint() and use_latest:
            if is_val:
                f = self.get_checkpoint_file(is_val=True)
            else:
                f = self.get_checkpoint_file()
        if not f:
            self.logger.warning("No checkpoint found.")
            return {}

        self.logger.warning("Loading checkpoint from {}".format(f))
        checkpoint = torch.load(f, map_location=torch.device("cpu"))
        model = self.model
        model.load_state_dict(checkpoint.pop("model"))
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.warning("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        return checkpoint


    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        return os.path.exists(save_file)

    def get_checkpoint_file(self, is_val=False):
        if is_val:
            save_file = os.path.join(self.save_dir, self._best_checkpoint_name)
        else:
            save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        try:
            with open (save_file, 'r') as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename, is_last, is_best):
        if is_last:
            save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
            with open(save_file, "w") as f:
                f.write(last_filename)
        elif is_best:
            save_file = os.path.join(self.save_dir, self._best_checkpoint_name)
            with open(save_file, "w") as f:
                f.write(last_filename)