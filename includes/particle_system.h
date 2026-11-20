// stolen from https://github.com/TheCherno/OneHourParticleSystem

#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H



#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ParticleProps
{
	glm::vec2 Position;
	glm::vec2 Velocity, VelocityVariation;
	glm::vec4 ColorBegin, ColorEnd;
	float SizeBegin, SizeEnd, SizeVariation;
	float LifeTime = 1.0f;
};

class ParticleSystem
{
public:
	ParticleSystem(const std::string& vertexPath, const std::string& fragmentPath)
		: m_VertexPath(vertexPath), m_FragmentPath(fragmentPath)
	{
		m_ParticlePool.resize(1000);
	}

	void OnUpdate(float ts)
	{
		for (auto& particle : m_ParticlePool)
		{
			if (!particle.Active)
				continue;

			if (particle.LifeRemaining <= 0.0f)
			{
				particle.Active = false;
				continue;
			}

			particle.LifeRemaining -= ts;
			particle.Position += particle.Velocity * ts;
			particle.Rotation += 0.01f * ts;
		}
	}

	void OnRender(const glm::mat4& viewProj)
	{
		if (!m_QuadVA)
			Init();

		m_ParticleShader->use();
		glUniformMatrix4fv(m_ParticleShaderViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));

		for (auto& particle : m_ParticlePool)
		{
			if (!particle.Active)
				continue;

			// Fade away particles
			float life = particle.LifeRemaining / particle.LifeTime;
			glm::vec4 color = glm::mix(particle.ColorEnd, particle.ColorBegin, life);

			float size = glm::mix(particle.SizeEnd, particle.SizeBegin, life);

			// Render
			glm::mat4 transform = glm::translate(glm::mat4(1.0f), { particle.Position.x, particle.Position.y, 0.0f })
				* glm::rotate(glm::mat4(1.0f), particle.Rotation, { 0.0f, 0.0f, 1.0f })
				* glm::scale(glm::mat4(1.0f), { size, size, 1.0f });
			glUniformMatrix4fv(m_ParticleShaderTransform, 1, GL_FALSE, glm::value_ptr(transform));
			glUniform4fv(m_ParticleShaderColor, 1, glm::value_ptr(color));
			glBindVertexArray(m_QuadVA);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
		}
		glBindVertexArray(0);
	}

	void Emit(const ParticleProps& particleProps)
	{
		Particle& particle = m_ParticlePool[m_PoolIndex];
		particle.Active = true;
		particle.Position = particleProps.Position;
		particle.Rotation = RandomFloat() * 2.0f * glm::pi<float>();

		// Velocity
		particle.Velocity = particleProps.Velocity;
		particle.Velocity.x += particleProps.VelocityVariation.x * (RandomFloat() - 0.5f);
		particle.Velocity.y += particleProps.VelocityVariation.y * (RandomFloat() - 0.5f);

		// Color
		particle.ColorBegin = particleProps.ColorBegin;
		particle.ColorEnd = particleProps.ColorEnd;

		particle.LifeTime = particleProps.LifeTime;
		particle.LifeRemaining = particleProps.LifeTime;
		particle.SizeBegin = particleProps.SizeBegin + particleProps.SizeVariation * (RandomFloat() - 0.5f);
		particle.SizeEnd = particleProps.SizeEnd;

		m_PoolIndex = (m_PoolIndex == 0) ? (uint32_t)m_ParticlePool.size() - 1 : m_PoolIndex - 1;
	}

private:
	struct Particle
	{
		glm::vec2 Position;
		glm::vec2 Velocity;
		glm::vec4 ColorBegin, ColorEnd;
		float Rotation = 0.0f;
		float SizeBegin, SizeEnd;

		float LifeTime = 1.0f;
		float LifeRemaining = 0.0f;

		bool Active = false;
	};
	std::vector<Particle> m_ParticlePool;
	uint32_t m_PoolIndex = 999;
	uint32_t m_RandomState = 0x9E3779B9u;

	unsigned int m_QuadVA = 0;
	std::string m_VertexPath, m_FragmentPath;
	std::unique_ptr<Shader> m_ParticleShader;
	int m_ParticleShaderViewProj, m_ParticleShaderTransform, m_ParticleShaderColor;

	// [0, 1) xorshift, replaces Random::Float()
	float RandomFloat()
	{
		m_RandomState ^= m_RandomState << 13;
		m_RandomState ^= m_RandomState >> 17;
		m_RandomState ^= m_RandomState << 5;
		return (float)(m_RandomState >> 8) * (1.0f / 16777216.0f);
	}

	void Init()
	{
		float vertices[] = {
			 -0.5f, -0.5f, 0.0f,
			  0.5f, -0.5f, 0.0f,
			  0.5f,  0.5f, 0.0f,
			 -0.5f,  0.5f, 0.0f
		};

		glGenVertexArrays(1, &m_QuadVA);
		glBindVertexArray(m_QuadVA);

		GLuint quadVB, quadIB;
		glGenBuffers(1, &quadVB);
		glBindBuffer(GL_ARRAY_BUFFER, quadVB);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

		uint32_t indices[] = {
			0, 1, 2, 2, 3, 0
		};

		glGenBuffers(1, &quadIB);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadIB);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glBindVertexArray(0);

		m_ParticleShader = std::make_unique<Shader>(m_VertexPath.c_str(), m_FragmentPath.c_str());
		m_ParticleShaderViewProj = glGetUniformLocation(m_ParticleShader->ID, "viewProj");
		m_ParticleShaderTransform = glGetUniformLocation(m_ParticleShader->ID, "transform");
		m_ParticleShaderColor = glGetUniformLocation(m_ParticleShader->ID, "color");
	}
};

#endif
