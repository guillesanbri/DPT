import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        check_parameters = ["scratch.layer1_rn.weight", "scratch.layer2_rn.weight",
                            "pretrained.model.patch_embed.proj.weight"]

        for check_param in check_parameters:
            if self.state_dict()[check_param].shape != parameters[check_param].shape:
                print(f"[I] Removing {check_param} from the parameters dict as its shape does not correspond to the model.")
                del parameters[check_param]

        self.load_state_dict(parameters, strict=False)
