"""GraphMixer implementation for OpenDG.

This module implements the GraphMixer model based on the DygLib implementation,
adapted for the OpenDG framework.

GraphMixer is a temporal graph learning model that combines graph neural networks
with MLP-Mixer architecture. It's designed to learn from dyanmic graphs where 
interactions between nodes happen over time. 
"""


from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from opendg.nn.time_encoding import Time2Vec


class FeedForwardNet(nn.Module):
    """Two-layered MLP with GELU activation function."""

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        Initialize the FeedForwardNet.
        
        Args:
            input_dim: Dimension of input
            dim_expansion_factor: Dimension expansion factor
            dropout: Dropout rate
        """
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed forward network forward pass.
        
        Args:
            x: Input tensor of shape (*, input_dim)
            
        Returns:
            Output tensor of the same shape as input
        """
        return self.ffn(x)


class MLPMixer(nn.Module):
    """MLP Mixer for token and channel mixing."""

    def __init__(
        self, 
        num_tokens: int, 
        num_channels: int, 
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0, 
        dropout: float = 0.0
    ):
        """
        Initialize the MLPMixer.
        
        Args:
            num_tokens: Number of tokens
            num_channels: Number of channels
            token_dim_expansion_factor: Dimension expansion factor for tokens
            channel_dim_expansion_factor: Dimension expansion factor for channels
            dropout: Dropout rate
        """
        super().__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(
            input_dim=num_tokens, 
            dim_expansion_factor=token_dim_expansion_factor,
            dropout=dropout
        )

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(
            input_dim=num_channels, 
            dim_expansion_factor=channel_dim_expansion_factor,
            dropout=dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        MLP mixer forward pass to compute over tokens and channels.
        
        Args:
            input_tensor: Tensor of shape (batch_size, num_tokens, num_channels)
            
        Returns:
            Output tensor of shape (batch_size, num_tokens, num_channels)
        """
        # Mix tokens
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        output_tensor = hidden_tensor + input_tensor

        # Mix channels
        hidden_tensor = self.channel_norm(output_tensor)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        output_tensor = hidden_tensor + output_tensor

        return output_tensor


class GraphMixer(nn.Module):
    """GraphMixer model for temporal graph learning."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int, 
        num_tokens: int, 
        num_layers: int = 2, 
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0, 
        dropout: float = 0.1
    ):
        """
        Initialize the GraphMixer model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            time_feat_dim: Dimension of time features (encodings)
            num_tokens: Number of tokens (neighbors to sample)
            num_layers: Number of MLP Mixer layers
            token_dim_expansion_factor: Dimension expansion factor for tokens
            channel_dim_expansion_factor: Dimension expansion factor for channels
            dropout: Dropout rate
        """
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_channels = edge_feat_dim

        # Time encoder - using OpenDG's Time2Vec
        self.time_encoder = Time2Vec(time_dim=time_feat_dim)
        
        # Projection layer to combine edge features and time features
        self.projection_layer = nn.Linear(edge_feat_dim + time_feat_dim, self.num_channels)

        # MLP Mixer layers
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(
                num_tokens=num_tokens, 
                num_channels=self.num_channels,
                token_dim_expansion_factor=token_dim_expansion_factor,
                channel_dim_expansion_factor=channel_dim_expansion_factor, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(
            in_features=self.num_channels + node_feat_dim, 
            out_features=node_feat_dim
        )

    def forward(
        self, 
        node_feats: torch.Tensor,
        neighbor_edge_feats: torch.Tensor,
        neighbor_time_diffs: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the GraphMixer model.
        
        Args:
            node_feats: Node features of shape (batch_size, node_feat_dim)
            neighbor_edge_feats: Edge features of neighbors, shape (batch_size, num_tokens, edge_feat_dim)
            neighbor_time_diffs: Time differences to neighbors, shape (batch_size, num_tokens)
            neighbor_mask: Mask for valid neighbors, shape (batch_size, num_tokens)
            
        Returns:
            Node embeddings of shape (batch_size, node_feat_dim)
        """
        # Encode time differences
        time_feats = self.time_encoder(neighbor_time_diffs.unsqueeze(-1))
        
        # Zero out time features for padding
        time_feats = time_feats * neighbor_mask.unsqueeze(-1)
        
        # Combine edge features and time features
        combined_features = torch.cat([neighbor_edge_feats, time_feats], dim=-1)
        combined_features = self.projection_layer(combined_features)
        
        # Apply MLP Mixer layers
        for mlp_mixer in self.mlp_mixers:
            combined_features = mlp_mixer(combined_features)
            
        # Aggregate neighbor features (mean pooling with mask)
        neighbor_contribution = torch.sum(
            combined_features * neighbor_mask.unsqueeze(-1), dim=1
        ) / torch.sum(neighbor_mask, dim=1, keepdim=True).clamp(min=1.0)
        
        # Combine with node features
        output = self.output_layer(torch.cat([neighbor_contribution, node_feats], dim=1))
        
        return output
        
    def compute_temporal_embeddings(
        self,
        node_feats: torch.Tensor,
        neighbor_node_feats: torch.Tensor,
        neighbor_edge_feats: torch.Tensor,
        neighbor_time_diffs: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal embeddings for nodes given their neighbors.
        
        Args:
            node_feats: Node features of shape (batch_size, node_feat_dim)
            neighbor_node_feats: Node features of neighbors, shape (batch_size, num_tokens, node_feat_dim)
            neighbor_edge_feats: Edge features of neighbors, shape (batch_size, num_tokens, edge_feat_dim)
            neighbor_time_diffs: Time differences to neighbors, shape (batch_size, num_tokens)
            neighbor_mask: Mask for valid neighbors, shape (batch_size, num_tokens)
            
        Returns:
            Node embeddings of shape (batch_size, node_feat_dim)
        """
        # Get neighbor contributions through MLP Mixer
        neighbor_contribution = self.forward(
            node_feats, 
            neighbor_edge_feats, 
            neighbor_time_diffs, 
            neighbor_mask
        )
        
        # Combine with original node features
        return neighbor_contribution + node_feats