#include "camera.h"
#include "imgui.h"
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
#include <palm_detector.h>
#include <hand_pose_detector.h>
#include <particle_system.h>

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

std::array<bool,5> fingerStates(const HandPose& h) {
  auto d = [&](int a, int b){ return cv::norm(h.landmarks[a] - h.landmarks[b]); };
  std::array<bool,5> up;
  up[0] = d(4, 17) > d(3, 17);                  // thumb
  const int tips[] = {8, 12, 16, 20}, pips[] = {6, 10, 14, 18};
  for (int i = 0; i < 4; ++i)
      up[i+1] = d(tips[i], 0) > d(pips[i], 0);  // other fingers
  return up;
}

std::vector<std::string> detectedGestures;

std::string classifyGesture(const HandPose& h) {
  auto f = fingerStates(h);   // {thumb, index, middle, ring, pinky}
  using A = std::array<bool,5>;
  if (f == A{0,0,0,0,0}) return "Fist";
  if (f == A{1,1,1,1,1}) return "Open Palm";
  if (f == A{0,1,0,0,0}) return "Pointing";
  if (f == A{0,1,1,0,0}) return "Victory";
  if (f == A{1,0,0,0,0}) return "Thumb";
  if (f == A{1,1,0,0,1}) return "I Love You";
  if (f == A{0,1,0,0,1}) return "Rock";
  return "Unknown";
}

int particlesCount  =0;
std::string prevGesture = "";

  void DrawMenu() {
    ImGui::Begin("Menu");
    ImGui::Text("window : %.0f x %.0f", (float)windowWidth, (float)windowHeight);
    for (size_t i = 0; i < detectedGestures.size(); ++i){
        ImGui::Text("Hand %zu: %s", i + 1, detectedGestures[i].c_str());
        ImGui::Text ("Particles Count: %d", particlesCount);
    }
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

  // Initialize webcam
  if (!webCam.open(0)) {
      std::cerr << "Failed to open webcam\n";
      return -1;
  }

  webCam.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  webCam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  std::cout << "Webcam initialized\n";

  PalmDetector palmDetector(std::string(logl_root) +"/models/palm_detection_mediapipe_2023feb.onnx");

  HandPoseDetector handPoseDetector(std::string(logl_root) +"/models/handpose_estimation_mediapipe_2023feb.onnx");

  unsigned int landmarkVAO, landmarkVBO;
  glGenVertexArrays(1, &landmarkVAO);
  glGenBuffers(1, &landmarkVBO);

  glBindVertexArray(landmarkVAO);
  glBindBuffer(GL_ARRAY_BUFFER, landmarkVBO);

  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindVertexArray(0);

  Shader landmarkShader(
      (std::string(logl_root) + "/src/shaders/landmarks.vs").c_str(),
      (std::string(logl_root) + "/src/shaders/landmarks.fs").c_str()
  );

  std::vector<float> landmarkPoints;
  std::vector<float> landmarkLines;
  
  ParticleSystem particleSystem(
      std::string(logl_root) + "/src/shaders/particle.vs",
      std::string(logl_root) + "/src/shaders/particle.fs"
  );

  ParticleProps pointerParticle;
  pointerParticle.Velocity          = { 0.0f, 0.0f };
  pointerParticle.VelocityVariation = { 0.6f, 0.6f };          // spread, in NDC units/sec
  pointerParticle.ColorBegin        = { 1.0f, 0.6f, 0.1f, 1.0f };
  pointerParticle.ColorEnd          = { 0.6f, 0.1f, 0.9f, 0.0f };
  pointerParticle.SizeBegin         = 0.04f;
  pointerParticle.SizeVariation     = 0.02f;
  pointerParticle.SizeEnd           = 0.0f;
  pointerParticle.LifeTime          = 0.8f;
  
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
   
    landmarkPoints.clear();
    landmarkLines.clear();
    detectedGestures.clear();

    cv::Mat cameraFrame;
    webCam >> cameraFrame;

    if (!cameraFrame.empty()) {
        auto palms = palmDetector.detect(cameraFrame);

        for (const auto& palm : palms) {
            try {
                HandPose hand = handPoseDetector.detect(cameraFrame, palm);
                std::string gesture = classifyGesture(hand);
                detectedGestures.push_back(gesture);

                // Convert camera pixels to OpenGL NDC coordinates.
                auto toNDC = [&](const cv::Point3f& p) {
                    float x = 2.0f * p.x / cameraFrame.cols - 1.0f;
                    float y = 1.0f - 2.0f * p.y / cameraFrame.rows;
                    return glm::vec2(x, y);
                };

                // Emit particles from the index fingertip while pointing.
                if (gesture == "Pointing") {
                    glm::vec2 tip = toNDC(hand.landmarks[8]);   // 8 = index fingertip
                    float aspect = (float)windowWidth / (float)windowHeight;
                    pointerParticle.Position = { tip.x * aspect, tip.y };
                    pointerParticle.VelocityVariation = { 0.6f, 0.6f };  
                    pointerParticle.SizeVariation     = 0.02f;
                    pointerParticle.SizeBegin         = 0.04f;
                    for (int i = 0; i < 5; ++i)
                        particleSystem.Emit(pointerParticle);
                }


                if (gesture == "Fist" || gesture == "Thumb" ) {
                    particlesCount++;
                }

                if (gesture == "Open Palm" && (prevGesture == "Fist" || prevGesture == "Thumb") && particlesCount > 0) {
                    glm::vec2 c = toNDC(hand.landmarks[9]);   // 9 = middle-finger base
                    float aspect = (float)windowWidth / (float)windowHeight;
                    pointerParticle.Position = { c.x * aspect, c.y };
                    pointerParticle.VelocityVariation = { 2.0f, 2.0f };
                    pointerParticle.SizeVariation     = 0.04f;
                    pointerParticle.SizeBegin = 0.1f;
                    for (int i = 0; i < particlesCount; ++i)
                        particleSystem.Emit(pointerParticle);
                }

                if (gesture != "Unknown") {
                    if (gesture != "Fist" && gesture != "Thumb" )
                        particlesCount = 0;
                    prevGesture = gesture;   
                }

                // Connect the joints into a hand skeleton.
                const int connections[][2] = {
                    {0,1}, {1,2}, {2,3}, {3,4},
                    {0,5}, {5,6}, {6,7}, {7,8},
                    {0,9}, {9,10}, {10,11}, {11,12},
                    {0,13}, {13,14}, {14,15}, {15,16},
                    {0,17}, {17,18}, {18,19}, {19,20},
                    {5,9}, {9,13}, {13,17}
                };

                for (const auto& edge : connections) {
                    glm::vec2 a = toNDC(hand.landmarks[edge[0]]);
                    glm::vec2 b = toNDC(hand.landmarks[edge[1]]);

                    landmarkLines.insert(landmarkLines.end(),{a.x, a.y, b.x, b.y});
                }
            }
            catch (const std::runtime_error&) {
                // Skip invalid or low-confidence detections.
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
      
    glDisable(GL_DEPTH_TEST);

    landmarkShader.use();
    landmarkShader.setVec3("pointColor", glm::vec3(0.0f, 1.0f, 0.0f));

    glBindVertexArray(landmarkVAO);
    glBindBuffer(GL_ARRAY_BUFFER, landmarkVBO);

    // Upload and draw skeleton lines.
    if (!landmarkLines.empty()) {
        
        glBufferData(GL_ARRAY_BUFFER, landmarkLines.size() * sizeof(float), landmarkLines.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, landmarkLines.size() / 2);
    }

    // Upload and draw landmark points.
    if (!landmarkPoints.empty()) {
        
        glBufferData(GL_ARRAY_BUFFER,landmarkPoints.size() * sizeof(float), landmarkPoints.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_POINTS, 0, landmarkPoints.size() / 2);
    }

    glBindVertexArray(0);

    // Particles share the landmarks' 2D NDC space, aspect-corrected so quads stay square.
    float aspect = (float)windowWidth / (float)windowHeight;
    glm::mat4 particleViewProj = glm::ortho(-aspect, aspect, -1.0f, 1.0f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    particleSystem.OnUpdate(deltaTime);
    particleSystem.OnRender(particleViewProj);
    glDisable(GL_BLEND);

    Gui::EndFrame();
    
    glfwSwapBuffers(window);

    // check and call events
    glfwPollEvents();

    cv::waitKey(1);
  }

  Gui::Shutdown();
 
  webCam.release();
  glfwTerminate();

  return 0;
}

GLFWwindow *startGLFW(int width, int height, const char *title, GLFWframebuffersizefun fb_cb, GLFWcursorposfun mouse_cb, GLFWscrollfun scroll_cb) {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
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
