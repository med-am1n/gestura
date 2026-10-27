#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

#include <glad/glad.h>


inline void createSphere(float rez, float r, std::vector<float>& vertices, unsigned int &vbo, unsigned int &vao){
 
  const double PI = std::acos(-1.0);
  
  vertices.clear();

  for (int i = 0; i < rez; i++) {
        float lon = -PI + (2.0f * PI * i) / (rez - 1);

        for (int j = 0; j < rez; j++) {
            float lat = -PI / 2.0f + (PI * j) / (rez - 1);

            float x = r * cos(lat) * cos(lon);
            float y = r * cos(lat) * sin(lon);
            float z = r * sin(lat);

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, vertices.size()* sizeof(float), vertices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3* sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindVertexArray(0);
  
}

inline void createCube(float size, std::vector<float>& vertices, unsigned int& vbo, unsigned int& vao)
{
    float s = size / 2.0f;

    vertices = {
        // Front
        -s,-s,s,  s,-s,s,  s,s,s,
        -s,-s,s,  s,s,s,  -s,s,s,

        // Back
        s,-s,-s,  -s,-s,-s,  -s,s,-s,
        s,-s,-s,  -s,s,-s,  s,s,-s,

        // Left
        -s,-s,-s,  -s,-s,s,  -s,s,s,
        -s,-s,-s,  -s,s,s,  -s,s,-s,

        // Right
        s,-s,s,  s,-s,-s,  s,s,-s,
        s,-s,s,  s,s,-s,  s,s,s,

        // Top
        -s,s,s,  s,s,s,  s,s,-s,
        -s,s,s,  s,s,-s,  -s,s,-s,

        // Bottom
        -s,-s,-s,  s,-s,-s,  s,-s,s,
        -s,-s,-s,  s,-s,s,  -s,-s,s
    };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

inline void createPlane(float size, std::vector<float>& vertices, unsigned int& vbo, unsigned int& vao)
{
    float s = size / 2.0f;

    vertices = {
        -s,-s,0,  s,-s,0,  s,s,0,
        -s,-s,0,  s,s,0,  -s,s,0
    };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

inline void createCylinder(float rez, float r, float h, std::vector<float>& vertices, unsigned int& vbo, unsigned int& vao)
{
    const float PI = std::acos(-1.0f);
    int n = std::max(3, (int)rez);

    float zTop = h / 2.0f;
    float zBottom = -h / 2.0f;

    vertices.clear();

    for (int i = 0; i < n; i++) {
        float a = 2.0f * PI * i / n;
        float b = 2.0f * PI * (i + 1) / n;

        float x1 = r * cos(a), y1 = r * sin(a);
        float x2 = r * cos(b), y2 = r * sin(b);

        vertices.insert(vertices.end(), {
            // Side
            x1,y1,zBottom, x2,y2,zBottom, x2,y2,zTop,
            x1,y1,zBottom, x2,y2,zTop,    x1,y1,zTop,

            // Top cap
            0,0,zTop, x1,y1,zTop, x2,y2,zTop,

            // Bottom cap
            0,0,zBottom, x2,y2,zBottom, x1,y1,zBottom
        });
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}


inline void createCone(float rez, float r, float h, std::vector<float>& vertices, unsigned int& vbo, unsigned int& vao)
{
    const float PI = std::acos(-1.0f);
    int n = std::max(3, (int)rez);

    float zBase = -h / 2.0f;
    float zTip = h / 2.0f;

    vertices.clear();

    for (int i = 0; i < n; i++) {
        float a = 2.0f * PI * i / n;
        float b = 2.0f * PI * (i + 1) / n;

        float x1 = r * cos(a), y1 = r * sin(a);
        float x2 = r * cos(b), y2 = r * sin(b);

        vertices.insert(vertices.end(), {
            // Side
            x1,y1,zBase, x2,y2,zBase, 0,0,zTip,

            // Base cap
            0,0,zBase, x2,y2,zBase, x1,y1,zBase
        });
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

inline void createTorus(float rez, float R, float r, std::vector<float>& vertices, unsigned int& vbo, unsigned int& vao)
{
    const float PI = std::acos(-1.0f);
    int n = std::max(3, (int)rez);

    vertices.clear();

    for (int i = 0; i < n; i++) {
        float u1 = 2.0f * PI * i / n;
        float u2 = 2.0f * PI * (i + 1) / n;

        for (int j = 0; j < n; j++) {
            float v1 = 2.0f * PI * j / n;
            float v2 = 2.0f * PI * (j + 1) / n;

            auto point = [&](float u, float v) {
                return std::array<float, 3>{
                    (R + r * cos(v)) * cos(u),
                    (R + r * cos(v)) * sin(u),
                    r * sin(v)
                };
            };

            auto p1 = point(u1, v1);
            auto p2 = point(u2, v1);
            auto p3 = point(u2, v2);
            auto p4 = point(u1, v2);

            vertices.insert(vertices.end(), {
                p1[0],p1[1],p1[2],
                p2[0],p2[1],p2[2],
                p3[0],p3[1],p3[2],

                p1[0],p1[1],p1[2],
                p3[0],p3[1],p3[2],
                p4[0],p4[1],p4[2]
            });
        }
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}
