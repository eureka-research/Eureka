camera_property = """
# Camera Properties 
self.cameras = []
self.camera_props = gymapi.CameraProperties()
self.camera_props.width = 256
self.camera_props.height = 256
self.camera_props.enable_tensors = True
"""

camera_sensor = """
# Camera Sensor
camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
local_pos = gymapi.Vec3(1., -1.0, 1.0)
target_pos = gymapi.Vec3(0., 0., 0.)
self.gym.set_camera_location(camera_handle, env_ptr, local_pos, target_pos)
self.cameras.append(camera_handle)
"""

camera_render = """
# Camera Render
camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
camera_rgba_image = torch_rgba_tensor.cpu().numpy()
self.img_buf = camera_rgba_image
"""