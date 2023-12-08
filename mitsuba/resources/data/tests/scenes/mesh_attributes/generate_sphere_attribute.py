import numpy as np

import mitsuba
mitsuba.set_variant("scalar_rgb")

from mitsuba.core.xml import load_string

m = load_string("""
    <shape type="ply" version="2.0.0">
        <string name="filename" value="meshes/sphere.ply"/>
    </shape>
""")

attribute = m.add_attribute("face_weight", 1)
attribute[:] = np.random.rand(m.face_count())

attribute = m.add_attribute("face_color", 3)
attribute[:] = np.random.rand(3 * m.face_count())

attribute = m.add_attribute("vertex_color", 3)
attribute[:] = np.random.rand(3 * m.vertex_count())

m.write_ply("meshes/sphere_attribute.ply")
