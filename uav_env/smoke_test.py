from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
w = World()
w.scene.add_default_ground_plane()
w.scene.add(DynamicCuboid(prim_path="/World/cube", position=np.array([0, 0, 1.0]), size=0.5, color=np.array([1.0, 0.2, 0.2])))
cam = rep.create.camera(position=(3, 3, 2), look_at=(0, 0, 0.5))
rp = rep.create.render_product(cam, (640, 480))
rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
w.reset()
for i in range(60):
    w.step(render=True)
img = rgb.get_data()
print("SMOKE_IMG", img.shape, img.mean())
from PIL import Image
Image.fromarray(img[..., :3]).save("smoke.png")
print("SMOKE_OK")
app.close()
