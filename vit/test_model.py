import torch

from model import MLP, EncoderBlock, MultiHeadSelfAttention, PatchEmbedding, ViT

torch.set_num_threads(2)


def test_patch_embedding_shape():
    embed = PatchEmbedding(image_size=28, patch_size=4, in_channels=1, dim=32)
    out = embed(torch.randn(2, 1, 28, 28))
    assert embed.num_patches == 49
    assert out.shape == (2, 49, 32)


def test_patch_embedding_matches_linear_projection():
    """The Conv2d patch embedding equals flattening each patch and multiplying by E."""
    P, C, D = 4, 1, 8
    embed = PatchEmbedding(image_size=8, patch_size=P, in_channels=C, dim=D)
    x = torch.randn(1, C, 8, 8)
    patches = x.unfold(2, P, P).unfold(3, P, P)              # (1, C, 2, 2, P, P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 4, C * P * P)
    E = embed.proj.weight.reshape(D, -1)
    expected = patches @ E.T + embed.proj.bias
    torch.testing.assert_close(embed(x), expected, rtol=1e-5, atol=1e-5)


def test_attention_shapes_and_rows_sum_to_one():
    attn = MultiHeadSelfAttention(dim=32, num_heads=4)
    x = torch.randn(3, 10, 32)
    out = attn(x)
    assert out.shape == (3, 10, 32)
    assert attn.last_attn.shape == (3, 4, 10, 10)
    torch.testing.assert_close(attn.last_attn.sum(-1), torch.ones(3, 4, 10))


def test_mlp_and_block_preserve_shape():
    x = torch.randn(2, 17, 32)
    assert MLP(32, 64)(x).shape == x.shape
    assert EncoderBlock(32, num_heads=4, mlp_dim=64)(x).shape == x.shape


def test_vit_forward_shape():
    model = ViT(image_size=28, patch_size=4, in_channels=1, num_classes=10,
                dim=32, depth=2, num_heads=4, mlp_dim=64)
    logits = model(torch.randn(5, 1, 28, 28))
    assert logits.shape == (5, 10)
    maps = model.attention_maps()
    assert len(maps) == 2
    assert maps[0].shape == (5, 4, 50, 50)


def test_vit_rgb_input():
    model = ViT(image_size=32, patch_size=8, in_channels=3, num_classes=100,
                dim=48, depth=1, num_heads=3, mlp_dim=96)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)
    assert model.pos_embed.shape == (1, 17, 48)


def test_vit_backward():
    model = ViT(dim=32, depth=2, num_heads=4, mlp_dim=64)
    loss = model(torch.randn(4, 1, 28, 28)).sum()
    loss.backward()
    assert model.cls_token.grad is not None
    assert model.pos_embed.grad is not None
    assert model.patch_embed.proj.weight.grad is not None
