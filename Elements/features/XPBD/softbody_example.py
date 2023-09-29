

import time
import numpy as np
import Elements.pyECSS.math_utilities as util
from Elements.pyECSS.Entity import Entity
from Elements.pyECSS.Component import BasicTransform, Camera, RenderMesh
from Elements.pyECSS.System import TransformSystem, CameraSystem
from Elements.pyGLV.GL.GameObject import GameObject
from Elements.pyGLV.GL.Scene import Scene
from Elements.pyGLV.GUI.Viewer import RenderGLStateSystem, ImGUIecssDecorator
from Elements.pyGLV.GL.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from Elements.pyGLV.GL.VertexArray import VertexArray
from OpenGL.GL import GL_LINES
import OpenGL.GL as gl
from Elements.utils.terrain import generateTerrain
from Elements.features.XPBD.solver import *
from Elements.features.XPBD.particle import *

from Elements.definitions import MODEL_DIR

from Elements.utils.helper_function import displayGUI_text
example_description = \
"This example demonstrates the use of the XPBD solver. \n" \
"The solver can be used to simulate any kind of constrained object, like softbody, cloth, rope, etc. \n" \
"Here, a tetrahedron softbody is simulated. \n" \

#Light
Lposition = util.vec(-1, 1.5, 1.2) #uniform lightpos
Lambientcolor = util.vec(1.0, 1.0, 1.0) #uniform ambient color
Lcolor = util.vec(1.0,1.0,1.0)
Lintensity = 40.0


scene = Scene()    

# Creates a tetrahedron softbody and adds it to the solver
def addTetrahedronSoftbody(solver:Solver, position:List[float], compliance:float):
    tetrahedron_vertices = [
    np.add(np.array([ 1.0, 0.0, 0.0]), position),
    np.add(np.array([ 0.0, 1.0, 0.0]), position),
    np.add(np.array([ 0.0, 0.0, 1.0]), position),
    np.add(np.array([ 0.0, 0.0, 0.0]), position)
    ]

    tetrahedron_edges = [
        [ 0, 1],
        [ 0, 2],
        [ 0, 3],
        [ 1, 2],
        [ 1, 3],
        [ 2, 3]
    ]

    # Create particles at each vertex
    base_index = len(solver.particles)
    for i in range(4):
        pos = np.array(tetrahedron_vertices[i])
        solver.particles.append(Particle(pos, 1.0, False))

    # Create distance constraint at each edge
    for i in range(6):
        solver.add_distance_constraint(tetrahedron_edges[i][0] + base_index, tetrahedron_edges[i][1] + base_index, compliance)

    # Create volume constraint
    solver.add_volume_constraint([base_index, base_index + 1, base_index + 2, base_index + 3], compliance)

    # solver.particles[1].is_kinematic = True

def get_solver_vertices(solver:Solver):
    vertices = []
    for particle in solver.particles:
        vertices.append([particle.position[0], particle.position[1], particle.position[2], 1.0])
    return np.array(vertices)

def get_solver_indices(solver:Solver):
    indices = []
    for distance_constraint in solver.distance_constraints:
        indices.append(distance_constraint.particle_index_0)
        indices.append(distance_constraint.particle_index_1)
        indices.append(distance_constraint.particle_index_0)
    return np.array(indices)

def get_solver_colors(solver:Solver):
    colors = []
    for particle in solver.particles:
        colors.append([1.0, 1.0, 1.0, 1.0])
    return np.array(colors)

def main():
    # Scenegraph with Entities, Components
    rootEntity = scene.world.createEntity(Entity(name="RooT"))
    entityCam1 = scene.world.createEntity(Entity(name="Entity1"))
    scene.world.addEntityChild(rootEntity, entityCam1)
    trans1 = scene.world.addComponent(entityCam1, BasicTransform(name="Entity1_TRS", trs=util.translate(0,0,-8)))

    eye = util.vec(2.5, 2.5, -2.5)
    target = util.vec(0.0, 0.0, 0.0)
    up = util.vec(0.0, 1.0, 0.0)
    view = util.lookat(eye, target, up)
    projMat = util.perspective(50.0, 1.0, 1.0, 10.0)

    m = np.linalg.inv(projMat @ view)

    xpbd_solver:Solver = Solver([],[0, -9.81, 0], 10)

    entityCam2 = scene.world.createEntity(Entity(name="Entity_Camera"))
    scene.world.addEntityChild(entityCam1, entityCam2)
    trans2 = scene.world.addComponent(entityCam2, BasicTransform(name="Camera_TRS", trs=util.identity()))
    orthoCam = scene.world.addComponent(entityCam2, Camera(m, "orthoCam","Camera","500"))


    light_node = scene.world.createEntity(Entity(name="LightPos"))
    scene.world.addEntityChild(rootEntity, light_node)
    light_transform = scene.world.addComponent(light_node, BasicTransform(name="Light_TRS", trs=util.scale(1.0, 1.0, 1.0) ))
    # light_mesh = scene.world.addComponent(light_node, RenderMesh(name="Light_Mesh"))

    # Systems
    transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
    camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
    renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
    initUpdate = scene.world.createSystem(InitGLShaderSystem())

    # Add softbody to solver
    addTetrahedronSoftbody(xpbd_solver, [0, 5, 0], .03)


    # obj_to_import = MODEL_DIR / "LivingRoom" / "Lamp" / "Lamp.obj"
    # model_entity = GameObject.Spawn(scene, obj_to_import, "Lamp", rootEntity, util.translate(-0.2, 0.4, 0.0))


    # Light Visualization
    # a simple tetrahedron
    tetrahedron_vertices = np.array([
        [  1.0,  1.0,  1.0, 1.0 ], 
        [ -1.0, -1.0,  1.0, 1.0 ], 
        [ -1.0,  1.0, -1.0, 1.0 ], 
        [  1.0, -1.0, -1.0, 1.0 ]
    ],dtype=np.float32) 
    tetrahedron_colors = np.array([
        [  1.0,  0.0,  0.0, 1.0 ],
        [  0.0,  1.0,  0.0, 1.0 ],  
        [  0.0,  0.0,  1.0, 1.0 ], 
        [  1.0,  1.0,  1.0, 1.0 ]
    ])
    tetrahedron_indices = np.array([0, 2, 1, 0, 1, 3, 2, 3, 1, 3, 2, 0])

    # light_mesh.vertex_attributes.append(tetrahedron_vertices)
    # light_mesh.vertex_attributes.append(tetrahedron_colors)
    # light_mesh.vertex_index.append(tetrahedron_indices)
    light_vArray = scene.world.addComponent(light_node, VertexArray())
    light_shader_decorator = scene.world.addComponent(light_node, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))




    # Generate terrain
    vertexTerrain, indexTerrain, colorTerrain= generateTerrain(size=4,N=20)
    # Add terrain
    terrain = scene.world.createEntity(Entity(name="terrain"))
    scene.world.addEntityChild(rootEntity, terrain)
    terrain_trans = scene.world.addComponent(terrain, BasicTransform(name="terrain_trans", trs=util.identity()))
    terrain_mesh = scene.world.addComponent(terrain, RenderMesh(name="terrain_mesh"))
    terrain_mesh.vertex_attributes.append(vertexTerrain) 
    terrain_mesh.vertex_attributes.append(colorTerrain)
    terrain_mesh.vertex_index.append(indexTerrain)
    terrain_vArray = scene.world.addComponent(terrain, VertexArray(primitive=GL_LINES))
    terrain_shader = scene.world.addComponent(terrain, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))
    # terrain_shader.setUniformVariable(key='modelViewProj', value=mvpMat, mat4=True)

    ## ADD AXES ##
    #Colored Axes
    vertexAxes = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [1.5, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.5, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.5, 1.0]
    ],dtype=np.float32) 
    colorAxes = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ], dtype=np.float32)


    #index arrays for above vertex Arrays
    index = np.array((0,1,2), np.uint32) #simple triangle
    indexAxes = np.array((0,1,2,3,4,5), np.uint32) #3 simple colored Axes as R,G,B lines

    axes = scene.world.createEntity(Entity(name="axes"))
    scene.world.addEntityChild(rootEntity, axes)
    axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=  util.translate(0.0, 0.00001, 0.0))) #util.identity()
    axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))
    axes_mesh.vertex_index.append(indexAxes)
    axes_vArray = scene.world.addComponent(axes, VertexArray(primitive=gl.GL_LINES)) # note the primitive change
    axes_shader = scene.world.addComponent(axes, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

    # softbody visual
    softbody = scene.world.createEntity(Entity(name="softbody"))
    scene.world.addEntityChild(rootEntity, softbody)
    softbody_transform = scene.world.addComponent(softbody, BasicTransform(name="softbody_transform", trs=util.identity()))
    softbody_mesh = scene.world.addComponent(softbody, RenderMesh(name="softbody_mesh"))
    softbody_mesh.vertex_attributes= [get_solver_vertices(xpbd_solver), get_solver_colors(xpbd_solver)]
    softbody_mesh.vertex_index.append(get_solver_indices(xpbd_solver))
    softbody_v_array = scene.world.addComponent(softbody, VertexArray(primitive=gl.GL_LINES))
    softbody_shader = scene.world.addComponent(softbody, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

    # MAIN RENDERING LOOP
    running = True
    scene.init(imgui=True, windowWidth = 1200, windowHeight = 800, windowTitle = "Elements: Import wavefront .obj example", openGLversion = 4, customImGUIdecorator = ImGUIecssDecorator)

    # pre-pass scenegraph to initialise all GL context dependent geometry, shader classes
    # needs an active GL context
    scene.world.traverse_visit(initUpdate, scene.world.root)

    ################### EVENT MANAGER ###################

    eManager = scene.world.eventManager
    gWindow = scene.renderWindow
    gGUI = scene.gContext

    renderGLEventActuator = RenderGLStateSystem()


    eManager._subscribers['OnUpdateWireframe'] = gWindow
    eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
    eManager._subscribers['OnUpdateCamera'] = gWindow 
    eManager._actuators['OnUpdateCamera'] = renderGLEventActuator


    eye = util.vec(2.5, 2.5, 2.5)
    target = util.vec(0.0, 0.0, 0.0)
    up = util.vec(0.0, 1.0, 0.0)
    view = util.lookat(eye, target, up)
    projMat = util.perspective(50.0, 1200/800, 0.01, 100.0) ## WORKING 

    gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

    model_terrain_axes = util.translate(0.0,0.0,0.0)

    # Initialize mesh GL depended components
    # model_entity.initialize_gl(Lposition, Lcolor, Lintensity)

    # model_entity.transform_component.trs = util.scale(1.0, 1.0, 1.0)

    previous_time = time.time()

    while running:
        delta_time = time.time() - previous_time
        previous_time = time.time()
        running = scene.render()
        displayGUI_text(example_description)
        scene.world.traverse_visit(renderUpdate, scene.world.root)
        scene.world.traverse_visit_pre_camera(camUpdate, orthoCam)
        scene.world.traverse_visit(camUpdate, scene.world.root)
        scene.world.traverse_visit(transUpdate, scene.world.root)

        view =  gWindow._myCamera # updates view via the imgui
        # mvp_cube = projMat @ view @ model_cube
        light_shader_decorator.setUniformVariable(key="modelViewProj", value= projMat @ view @ (util.translate(Lposition[0], Lposition[1], Lposition[2]) @ util.scale(0.05, 0.05, 0.05)), mat4=True)
        mvp_terrain = projMat @ view @ terrain_trans.trs
        mvp_axes = projMat @ view @ axes_trans.trs
        axes_shader.setUniformVariable(key='modelViewProj', value=mvp_axes, mat4=True)
        softbody_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain, mat4=True)
        terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain, mat4=True)

        # Update solver 
        if(delta_time > 0.0):
            xpbd_solver.step(delta_time) # TODO: use dt from scene
            softbody_mesh.vertex_attributes= [get_solver_vertices(xpbd_solver), get_solver_colors(xpbd_solver)]


        # # Set Object Real Time Shader Data
        # for mesh_entity in model_entity.mesh_entities:
        #     # --- Set vertex shader data ---
        #     mesh_entity.shader_decorator_component.setUniformVariable(key='projection', value=projMat, mat4=True)
        #     mesh_entity.shader_decorator_component.setUniformVariable(key='view', value=view, mat4=True)
        #     mesh_entity.shader_decorator_component.setUniformVariable(key='model', value=mesh_entity.transform_component.l2world, mat4=True)
        #     # Calculate normal matrix
        #     normalMatrix = np.transpose(util.inverse(mesh_entity.transform_component.l2world))
        #     mesh_entity.shader_decorator_component.setUniformVariable(key='normalMatrix', value=normalMatrix, mat4=True)

        #     # --- Set fragment shader data ---
        #     # Camera position
        #     mesh_entity.shader_decorator_component.setUniformVariable(key='camPos', value=eye, float3=True)

        scene.render_post()
        
    scene.shutdown()


if __name__ == "__main__":
    main()
