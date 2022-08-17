import torch
from copy import deepcopy
import torch.nn as nn
from typing import Callable, Iterable, List, Optional, Union

def get_module_from_param_name(net, param_name):
	# TODO: Can we just use nn.get_submodule()?
	names = param_name.split(".")[:-1]
	# recursively go through modules
	for name in names:
		net = getattr(net, name)
	return net

def attach_masks_as_parameter(
    net: nn.Module,
    structured: bool,
    gradient_tie: bool,
    masks_init_values: Optional[List[torch.Tensor]] = None,
    override_forward: bool = True,
    make_weights_constants: bool = True,
	) -> List[torch.Tensor]:

    def masked_conv2d_fwd(self, x):
        return F.conv2d(
            x,
            self.weight * self.weight_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def masked_linear_fwd(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

    last_layer = None

    all_weight_masks = []

    if masks_init_values:
        masks_init_values = iter(masks_init_values)

    for name, layer in net.named_modules():

        if structured:
            if gradient_tie and (
                "downsample" in name or ("n_block" in name and "conv2" in name)
            ):
                # tie the weight mask of conv and downsample layer
                layer.weight_mask = last_layer.weight_mask
            else:
                # Same channels, 1 all other dimensions
                shape = [layer.weight.shape[0]] + [1] * (layer.weight.ndim - 1)
                layer.weight_mask = nn.Parameter(
                    torch.ones(shape, device=layer.weight.device)
                )

        else:
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))

        if masks_init_values:
            layer.weight_mask.data[:] = next(masks_init_values)[:]

        all_weight_masks.append(layer.weight_mask)

        if make_weights_constants:
            layer.weight.requires_grad = False

        if "ds_block" in name or last_layer is None:
            last_layer = layer

        if override_forward:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(masked_conv2d_fwd, layer)
            elif isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(masked_linear_fwd, layer)
            else:
                raise TypeError

    return all_weight_masks

def pscore(net,device):
	device_orig = next(net.parameters()).device
	net_orig = net

	net = deepcopy(net)
	net.train()

	# Computing full meta gradients is pretty memory-intensive and therefore
	# better-suited for the CPU. We're (currently) only doing the inner-loop once anyway
	# so the speed-penalty is not that bad.
	#if method == "full":
	net = net.cpu()

	device = next(net.parameters()).device

	# after attach_masks_as_parameter() .parameters() will include masks too so let's
	# get params (and their names and modules) here
	detailed_params = [
		(n, get_module_from_param_name(net, n), p)
		for (n, p) in net.named_parameters()
	]
	#print('detail parameters-=====>', detailed_params)
	weight_masks = attach_masks_as_parameter(
	 	net,
	 	structured=False,
	 	gradient_tie=False,
	 	make_weights_constants=False,
	 	override_forward=False,
	)

	# scores = [torch.zeros_like(mask) for mask in weight_masks]