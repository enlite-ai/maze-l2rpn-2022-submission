"""Contains a policy network for unitary env."""
from collections import OrderedDict
from typing import List, Union, Sequence, Dict
from typing import Tuple

import torch
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.action_masking import ActionMaskingBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.flatten import FlattenBlock
from maze.perception.blocks.general.functional import FunctionalBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc
from torch import nn as nn


class BaseNet(nn.Module):
    """Feed forward base network.

    :param obs_shapes: The observation shape.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: The number of hidden units.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)],
                 hidden_units: List[int]):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.perception_dict = OrderedDict()

        # --- concatenate stacked node features if stacked observations are provided ---
        def concatenate_stack(x: torch.Tensor, apply_stacking: bool) -> torch.Tensor:
            """ Concatenate feature stack along last axis """
            if apply_stacking:
                return torch.cat([x[..., i, :] for i in range(x.shape[-2])], dim=-1)
            else:
                return x

        apply_stacking = len(self.obs_shapes['features']) == 2

        self.perception_dict['features_stacked'] = FunctionalBlock(
            in_keys='features', in_shapes=self.obs_shapes['features'], out_keys='features_stacked',
            func=lambda x: concatenate_stack(x, apply_stacking)
        )

        self.perception_dict['topology_stacked'] = FunctionalBlock(
            in_keys='topology', in_shapes=self.obs_shapes['topology'], out_keys='topology_stacked',
            func=lambda x: concatenate_stack(x, apply_stacking)
        )

        in_shapes = [self.obs_shapes['link_to_set_mask']]
        in_keys = ['link_to_set_mask']
        if 'already_selected_actions' in self.obs_shapes:
            in_shapes += [self.obs_shapes['already_selected_actions']]
            in_keys += ['already_selected_actions']

        self.perception_dict['mask_feat'] = ConcatenationBlock(
            in_keys=in_keys,
            in_shapes=in_shapes, out_keys='mask_feat', concat_dim=-1
        )

        # --- ff style model ---

        self.perception_dict['features_flat'] = FlattenBlock(
            in_keys='features_stacked', in_shapes=self.perception_dict['features_stacked'].out_shapes(),
            out_keys='features_flat', num_flatten_dims=1)

        self.perception_dict['topology_flat'] = FlattenBlock(
            in_keys='topology_stacked', in_shapes=self.perception_dict['topology_stacked'].out_shapes(),
            out_keys='topology_flat', num_flatten_dims=1)

        self.perception_dict['features_embed'] = DenseBlock(
            in_keys='features_flat', in_shapes=self.perception_dict['features_flat'].out_shapes(),
            out_keys='features_embed', hidden_units=hidden_units, non_lin=non_lin)

        self.perception_dict['topology_embed'] = DenseBlock(
            in_keys='topology_flat', in_shapes=self.perception_dict['topology_flat'].out_shapes(),
            out_keys='topology_embed', hidden_units=hidden_units, non_lin=non_lin)

        self.perception_dict['mask_embed'] = DenseBlock(
            in_keys='mask_feat', in_shapes=self.perception_dict['mask_feat'].out_shapes(),
            out_keys='mask_embed', hidden_units=hidden_units, non_lin=non_lin)

        in_shapes = self.perception_dict['features_embed'].out_shapes() + \
                    self.perception_dict['topology_embed'].out_shapes() + \
                    self.perception_dict['mask_embed'].out_shapes()
        self.perception_dict['hidden_out'] = ConcatenationBlock(
            in_keys=['features_embed', 'topology_embed', 'mask_embed'],
            in_shapes=in_shapes, out_keys='hidden_out', concat_dim=-1
        )

    def build_inference_block(self, out_keys: Union[str, List[str]]) -> InferenceBlock:
        """implementation of :class:`~maze.perception.blocks.inference.InferenceBlockBuilder` interface
        """
        in_keys = set(sum([block.in_keys for block in self.perception_dict.values()], []))
        in_keys = list(filter(lambda key: key in self.obs_shapes.keys(), in_keys))
        inference_block = InferenceBlock(
            in_keys=in_keys, out_keys=out_keys,
            in_shapes=[self.obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        return inference_block


class PolicyNet(BaseNet):
    """Policy network for unitary env.

    :param obs_shapes: The observation shape.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: The number of hidden units.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 action_logits_shapes: Dict[str, Tuple[int]],
                 non_lin: Union[str, type(nn.Module)],
                 hidden_units: List[int]):
        super().__init__(obs_shapes, non_lin, hidden_units)

        # add link prediction path
        if 'link_to_set' in action_logits_shapes:
            action_head_name = 'link_to_set'
            mask_name = "link_to_set_mask"
        else:
            action_head_name = 'action'
            mask_name = "action_mask"

        if action_head_name in action_logits_shapes:
            head_hidden_units = [lambda out_shape: int(out_shape[-1])]
            head_hidden_units = [func(self.perception_dict['hidden_out'].out_shapes()[0]) for func in head_hidden_units]

            self.perception_dict[f'{action_head_name}_net'] = DenseBlock(
                in_keys='hidden_out', in_shapes=self.perception_dict['hidden_out'].out_shapes(),
                out_keys=f'{action_head_name}_net', hidden_units=head_hidden_units, non_lin=non_lin)

            self.perception_dict[f'{action_head_name}_logits'] = LinearOutputBlock(
                in_keys=f'{action_head_name}_net',
                in_shapes=self.perception_dict[f'{action_head_name}_net'].out_shapes(),
                out_keys=f'{action_head_name}_logits', output_units=action_logits_shapes[action_head_name][0]
            )

            in_shapes = self.perception_dict[f'{action_head_name}_logits'].out_shapes() + \
                        [self.obs_shapes[mask_name]]
            self.perception_dict[f'{action_head_name}'] = ActionMaskingBlock(
                in_keys=[f'{action_head_name}_logits', mask_name], out_keys=f'{action_head_name}',
                in_shapes=in_shapes, num_actors=1, num_of_actor_actions=None
            )

        # Set up inference block
        self.perception_net = self.build_inference_block(list(action_logits_shapes.keys()))

        self.perception_net.apply(make_module_init_normc(1.0))
        for action_head_name in action_logits_shapes.keys():
            self.perception_dict[f'{action_head_name}'].apply(make_module_init_normc(0.01))

    def forward(self, xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param xx: input dict.
        :return: the computed output of the network.
        """
        return self.perception_net(xx)
