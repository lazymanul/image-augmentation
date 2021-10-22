import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from TextureLoader import load_texture
from ObjLoader import ObjLoader
import numpy as np
import cv2

def camera_calibration(sz):
    """ Simplified calibration routine """
    dZ = 500
    dx = dy = 870
    dX = dY = 129
    W = 4160
    H = 3120
    col, row = sz
    
    fxBase = dx * dZ / dX
    fyBase = dy * dZ / dY
    #print(fxBase, fyBase)

    fx = fxBase * col / W
    fy = fyBase * row / H
    K = np.diag([fx,fy,1])
    K[0,2] = 0.5 * col
    K[1,2] = 0.5 * row
    return K

def set_projection_from_camera(K, width, height):
    """ Set view from a camera calibration matrix. """
    fx = K[0,0]
    fy = K[1,1]
    fovy = 2 * np.arctan(0.5*height/fy) * 180 / np.pi
    aspect = (width*fy) / (height*fx)
    # define the near and far clipping planes
    near = 0.1
    far = 200.0
    # set perspective
    return pyrr.matrix44.create_perspective_projection_matrix(fovy, aspect, near, far)

def detect_markers(image, arucoType = cv2.aruco.DICT_ARUCO_ORIGINAL):
    """ Detect aruco markers on given image. """
    arucoType = cv2.aruco.DICT_ARUCO_ORIGINAL
    arucoDict = cv2.aruco.Dictionary_get(arucoType)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, 
        parameters=arucoParams)
    
    return zip(corners, ids)

def get_Rt(src, dst, K):    
    """ Compute [R|t] external camera matrix """
    distortion = np.zeros((4,1))
    ret, rvecs, tvecs = cv2.solvePnP(src, dst, K, distCoeffs=distortion)
    R, _ = cv2.Rodrigues(rvecs)
    
    return rvecs, tvecs, np.hstack((R, tvecs))

def get_transform_matrices(src, dst, K):
    """ Calculate model matrix for given image """    
    # convert coordinates from OpenCV-style to OpenGL
    conv = np.diag((1, -1, -1)) 
    rvecs, tvecs, Rt = get_Rt(src, dst, K)
    Rt = conv @ Rt
    
    R = pyrr.matrix44.create_from_matrix33(Rt[:3,:3]) 
    model = R.T @ pyrr.matrix44.create_from_translation(pyrr.Vector3(Rt[:3,3])) 
    return model

background_vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

out vec2 v_texture;

void main()
{
    gl_Position = vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec2 v_texture;

void main()
{    
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);
}
"""

width, height = 943, 708

# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(width, height, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

VAO = glGenVertexArrays(3)
VBO = glGenBuffers(3)

back_shader = compileProgram(compileShader(background_vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# VAO and VBO
vertices = [-1.0, -1.0, -1.0, 0.0, 0.0,
            -1.0,  1.0, -1.0, 0.0, 1.0,
             1.0,  1.0, -1.0, 1.0, 1.0,
            
            -1.0, -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, -1.0, 1.0, 1.0,]

vertices = np.array(vertices, dtype=np.float32)

# Background VAO
glBindVertexArray(VAO[0])
# Background Vertex Buffer Object
glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW) 

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
# load here the 3d meshes
model_indices, model_buffer = ObjLoader.load_model("data/cottage_blender.obj")

# model VAO
glBindVertexArray(VAO[1])
# model Vertex Buffer Object
glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
glBufferData(GL_ARRAY_BUFFER, model_buffer.nbytes, model_buffer, GL_STATIC_DRAW)

# model vertices
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, model_buffer.itemsize * 8, ctypes.c_void_p(0))
# model textures
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, model_buffer.itemsize * 8, ctypes.c_void_p(12))
# model normals
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, model_buffer.itemsize * 8, ctypes.c_void_p(20))
glEnableVertexAttribArray(2)


textures = glGenTextures(2)
load_texture("data/back_small.png", textures[0])
load_texture("data/cottage_diffuse.png", textures[1])

image = cv2.imread("data/back_small.png")
src = np.float32([[-9,0,-9], [9,0,-9], [9,0,9], [-9,0,9]]).reshape((-1,3))

K = camera_calibration((width, height))
projection = set_projection_from_camera(K, width, height)


markers = list(detect_markers(image))

(marker, markerID) = markers[3]
dst = marker.reshape((4,2))
model3 = get_transform_matrices(src, dst, K) 
view3 = np.eye(4)

(marker, markerID) = markers[0]
dst = marker.reshape((4,2))
model2 = get_transform_matrices(src, dst, K) 
view2 = np.eye(4)

glClearColor(0, 0.1, 0.1, 1)

glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)

    # draw the background
    glUseProgram(back_shader)
     
    glBindVertexArray(VAO[0])
    glBindTexture(GL_TEXTURE_2D, textures[0])    
    glDrawArrays(GL_TRIANGLES, 0, 6) 
    
    glEnable(GL_DEPTH_TEST)
    
    #draw the model (at 2 places)
    glUseProgram(shader)
    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)        

    glBindVertexArray(VAO[1])    
    glBindTexture(GL_TEXTURE_2D, textures[1])

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view3)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model3)
    glDrawArrays(GL_TRIANGLES, 0, len(model_indices))
    
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view2)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model2)
    glDrawArrays(GL_TRIANGLES, 0, len(model_indices))

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()
