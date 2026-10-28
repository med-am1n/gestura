#include "camera.h"
#include "imgui.h"
#include "onnxruntime_c_api.h"
#include "shader.h"
#include <GLFW/glfw3.h>
#include <cstddef>
#include <filesystem.h>
#include <glad/glad.h>
#include <iostream>
#include <mesh.h>
#include <ostream>
#include <stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>
#include <gui.h>
#include <shapes.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void window_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
GLFWwindow *startGLFW(int width, int height, const char *title, GLFWframebuffersizefun fb_cb, GLFWcursorposfun mouse_cb, GLFWscrollfun scroll_cb);
unsigned int LoadTexture(const char *path);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
int windowWidth = SCR_WIDTH;
int windowHeight = SCR_HEIGHT;
// camera
Camera camera(glm::vec3(0.0f, 1.0f, 3.0f));
float lastX = windowWidth / 2.0f;
float lastY = windowHeight / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;


void DrawMenu()
{
    ImGui::Begin("Menu");

    ImGui::Text("window : %.0f x %.0f", (float)windowWidth, (float)windowHeight); 
      
    ImGui::End();
}

cv::VideoCapture webCam;

int main() {

  std::cout << "Hello, World!" << std::endl;

  glfwInit();

  GLFWwindow *window = startGLFW(windowWidth, windowHeight, "Getsura", framebuffer_size_callback, mouse_callback, scroll_callback);

  if (!window)
    return -1;

  // initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
    std::cout << "Failed to initialize GLAD\n";
    return -1;
  }
 
  // Load onnx model
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
  Ort::SessionOptions session_options;
  Ort::Session session(env, (std::string(logl_root) + "/models/palm_detection_mediapipe_2023feb.onnx").c_str(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;

  size_t input_modes_size = session.GetInputCount();
  for(size_t i=0; i< input_modes_size; i++){
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
    std::cout << "Input Name "<<i << ": "<<input_name.get()<<std::endl;
  }

  size_t output_modes_size = session.GetOutputCount();
  for(size_t i=0; i< output_modes_size; i++){
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
    std::cout << "Output Name "<<i << ": "<<output_name.get()<<std::endl;
  }
 
  std::vector<const char*> input_names = {"input_1"};
  std::vector<const char*> output_names = {"Identity", "Identity_1"};
  std::vector<int64_t> input_shape = {1, 192, 192, 3};  
  

  // Initialize webcam
  if (!webCam.open(0)) {
      std::cerr << "Failed to open webcam\n";
      return -1;
  }

  webCam.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  webCam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  std::cout << "Webcam initialized\n";

  unsigned int vbo, vao;
  std::vector<float> vertices;
  Shader shader = Shader((std::string(logl_root) + "/src/shaders/shader.vs").c_str(), (std::string(logl_root) + "/src/shaders/shader.fs").c_str());

  glEnable(GL_DEPTH_TEST);
  
  // initialize Imgui
  Gui::Init(window);
  
  // render loop
  while (!glfwWindowShouldClose(window)) {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    
    // input
    processInput(window);
    
    Gui::BeginFrame();
   
    cv::Mat cameraFrame;
    if (webCam.isOpened()) {
        webCam >> cameraFrame;

        if (cameraFrame.empty()) {
            std::cerr << "Failed to capture camera frame\n";
        }

        // preprocessing the frmae
        cv::Mat resized;
        cv::resize(cameraFrame, resized, cv::Size(192, 192));

        // Convert BGR image to float32
        cv::Mat floatImage;
        resized.convertTo(floatImage, CV_32FC3);

        std::vector<float> inputTensorValues(1 * 3 * 192 * 192);

        const int width = 192;
        const int height = 192;
        const int planeSize = width * height;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                cv::Vec3f pixel = floatImage.at<cv::Vec3f>(y, x);

                int index = y * width + x;

                // OpenCV BGR order
                inputTensorValues[index] = pixel[0];
                inputTensorValues[planeSize + index] = pixel[1];
                inputTensorValues[2 * planeSize + index] = pixel[2];
            }
        }
    }
    
    DrawMenu();    

    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // rendering commands here

    // create transformations
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)windowWidth / (float)windowHeight, 0.1f, 100.0f);
    // camera
    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 model = glm::mat4(1.0f);
      
    shader.use();
    shader.setMat4("projection", projection);
    shader.setMat4("view", view);
    shader.setMat4("model", model);
    
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, vertices.size()/3);
    glBindVertexArray(0);

    Gui::EndFrame();
    
    glfwSwapBuffers(window);

    // check and call events
    glfwPollEvents();
  }

  Gui::Shutdown();
 
  webCam.release();
  glfwTerminate();

  return 0;
}

GLFWwindow *startGLFW(int width, int height, const char *title, GLFWframebuffersizefun fb_cb, GLFWcursorposfun mouse_cb, GLFWscrollfun scroll_cb) {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow *window = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!window) {
    std::cout << "Failed to create GLFW window\n";
    glfwTerminate();
    return nullptr;
  }

  glfwMakeContextCurrent(window);

  glfwSetFramebufferSizeCallback(window, fb_cb);
  glfwSetCursorPosCallback(window, mouse_cb);
  glfwSetScrollCallback(window, scroll_cb);
  glfwSetWindowSizeCallback(window, window_size_callback);
  return window;
}

unsigned int LoadTexture(const char *path) {
  unsigned int textureID;
  glGenTextures(1, &textureID);

  int width, height, nrChannels;
  unsigned char *data = stbi_load(path, &width, &height, &nrChannels, 0);

  if (data) {
    GLenum format;

    if (nrChannels == 1)
      format = GL_RED;
    else if (nrChannels == 3)
      format = GL_RGB;
    else if (nrChannels == 4)
      format = GL_RGBA;
    else
      format = GL_RGB;

    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
                 GL_UNSIGNED_BYTE, data);

    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
  } else {
    std::cout << "Failed to load texture: " << path << std::endl;
    stbi_image_free(data);
  }

  return textureID;
}

void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(FORWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(BACKWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(RIGHT, deltaTime);
}
// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

void window_size_callback(GLFWwindow *window, int width, int height) {
  windowWidth = width;
  windowHeight = height;
}
// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn) {
  // if (ImGui::GetIO().WantCaptureMouse) {
  //       firstMouse = true;
  //       return;
  //   }
  float xpos = static_cast<float>(xposIn);
  float ypos = static_cast<float>(yposIn);

  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
    if (firstMouse) {
      lastX = xpos;
      lastY = ypos;
      firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset =
        lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
  } else {
    firstMouse = true;
  }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
