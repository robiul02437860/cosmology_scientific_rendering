import torch
from gsplat.cuda import _wrapper

means = torch.tensor(
    [[[8.0, 8.0],
      [24.0, 8.0],
      [8.0, 24.0]]],
    device="cuda",
)

radii = torch.tensor(
    [[[2,2],
      [2,2],
      [2,2]]],
    dtype=torch.int32,
    device="cuda",
)

depths = torch.tensor(
    [[1.0,2.0,3.0]],
    device="cuda",
)

out = _wrapper.isect_tiles(
    means2d=means,
    radii=radii,
    depths=depths,
    tile_size=16,
    tile_width=2,
    tile_height=2,
    sort=True,
    segmented=False,
    packed=False,
)

print(type(out))
print(len(out))

for i, x in enumerate(out):
    print("="*60)
    print(i)
    print(type(x))
    print(x.dtype if torch.is_tensor(x) else "")
    print(x.shape if torch.is_tensor(x) else "")
    print(x)