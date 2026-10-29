#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "palm_detector.h"

struct HandPose {
    cv::Rect2f box;
    std::array<cv::Point3f, 21> landmarks;
    std::array<cv::Point3f, 21> worldLandmarks;
    float handedness = 0.0f; // 0 = Left, 1 = Right
    float confidence = 0.0f;
};

class HandPoseDetector {
public:
    static constexpr int INPUT_SIZE = 224;

    explicit HandPoseDetector(const std::string& modelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "HandPose"),
          session_(env_, modelPath.c_str(), Ort::SessionOptions{}) {
        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < session_.GetInputCount(); ++i) {
            auto name = session_.GetInputNameAllocated(i, allocator);
            inputNames_.emplace_back(name.get());
        }

        for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
            auto name = session_.GetOutputNameAllocated(i, allocator);
            outputNames_.emplace_back(name.get());
        }

        if (inputNames_.size() != 1 || outputNames_.size() != 4) {
            throw std::runtime_error(
                "Unexpected hand-pose model input/output count");
        }
    }

    // PalmDetection.box is x,y,width,height.
    // PalmDetection.landmarks contains the seven palm keypoints.
    HandPose detect(const cv::Mat& frame,const PalmDetection& palm) {
        if (frame.empty() || palm.landmarks.size() != 7) {
            throw std::invalid_argument("Invalid frame or palm landmarks");
        }

        // Build the format expected by the official MPHandPose algorithm:
        // [x1,y1,x2,y2, seven (x,y) palm landmarks].
        std::array<cv::Point2f, 7> palmPoints;
        for (int i = 0; i < 7; ++i) {
            palmPoints[i] = palm.landmarks[i];
        }

        cv::Rect2f b = palm.box;
        std::array<float, 18> palmData{};
        palmData[0] = b.x;
        palmData[1] = b.y;
        palmData[2] = b.x + b.width;
        palmData[3] = b.y + b.height;

        for (int i = 0; i < 7; ++i) {
            palmData[4 + i * 2] = palmPoints[i].x;
            palmData[5 + i * 2] = palmPoints[i].y;
        }

        cv::Point2i padBias;
        cv::Rect2f rotatedPalmBox;
        cv::Matx23f rotation;

        cv::Mat inputImage = preprocess(frame, palmData, padBias, rotatedPalmBox, rotation);

        // NHWC float32, RGB, normalized to [0,1].
        std::vector<float> inputValues(INPUT_SIZE * INPUT_SIZE * 3);

        for (int y = 0; y < INPUT_SIZE; ++y) {
            const auto* row = inputImage.ptr<cv::Vec3f>(y);
            for (int x = 0; x < INPUT_SIZE; ++x) {
                const int index = (y * INPUT_SIZE + x) * 3;
                inputValues[index]     = row[x][0];
                inputValues[index + 1] = row[x][1];
                inputValues[index + 2] = row[x][2];
            }
        }

        const std::array<int64_t, 4> shape = {1, INPUT_SIZE, INPUT_SIZE, 3};

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

        auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(), shape.data(), shape.size());

        std::vector<const char*> inputNamePtrs;
        std::vector<const char*> outputNamePtrs;

        for (const auto& s : inputNames_) inputNamePtrs.push_back(s.c_str());
        for (const auto& s : outputNames_) outputNamePtrs.push_back(s.c_str());

        auto outputs = session_.Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(), &inputTensor, 1, outputNamePtrs.data(), outputNamePtrs.size());

        // Official model output order:
        // landmarks, confidence, handedness, world landmarks.
        const float* screen = outputs[0].GetTensorData<float>();
        const float confidence = outputs[1].GetTensorData<float>()[0];
        const float handedness = outputs[2].GetTensorData<float>()[0];
        const float* world = outputs[3].GetTensorData<float>();

        if (confidence < confidenceThreshold_) {
            throw std::runtime_error("Hand pose confidence too low");
        }

        auto transform = decodeLandmarks(screen, world, confidence, handedness, rotatedPalmBox, padBias, rotation);

        return transform;
    }

    static void draw(cv::Mat& frame, const HandPose& hand) {
        static constexpr int connections[][2] = {
            {0,1},{1,2},{2,3},{3,4},
            {0,5},{5,6},{6,7},{7,8},
            {0,9},{9,10},{10,11},{11,12},
            {0,13},{13,14},{14,15},{15,16},
            {0,17},{17,18},{18,19},{19,20}
        };

        cv::rectangle(frame, hand.box, cv::Scalar(0,255,0), 2);

        for (const auto& connection : connections) {
            const auto& a = hand.landmarks[connection[0]];
            const auto& b = hand.landmarks[connection[1]];

            cv::line(frame,
                cv::Point(cvRound(a.x), cvRound(a.y)),
                cv::Point(cvRound(b.x), cvRound(b.y)),
                cv::Scalar(255,255,255), 2);
        }

        for (const auto& p : hand.landmarks) {
            cv::circle(frame, cv::Point(cvRound(p.x), cvRound(p.y)),3, cv::Scalar(0,0,255), -1);
        }

        cv::putText(frame, hand.handedness <= 0.5f ? "Left hand" : "Right hand",
            cv::Point(cvRound(hand.box.x), cvRound(hand.box.y)),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0,255,0), 1);
    }

    void setConfidenceThreshold(float value) {
        confidenceThreshold_ = value;
    }

private:
    Ort::Env env_;
    Ort::Session session_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    float confidenceThreshold_ = 0.8f;

    static cv::Mat cropAndPad(const cv::Mat& image, cv::Point2f& p0, cv::Point2f& p1, bool forRotation, cv::Point2i& bias) {

        cv::Point2f wh = p1 - p0;
        const float enlarge = forRotation ? 4.0f : 3.0f;

        cv::Point2f center = (p0 + p1) * 0.5f;
        cv::Point2f half = wh * (enlarge * 0.5f);

        p0 = center - half;
        p1 = center + half;

        int x0 = std::clamp(cvRound(p0.x), 0, image.cols);
        int y0 = std::clamp(cvRound(p0.y), 0, image.rows);
        int x1 = std::clamp(cvRound(p1.x), 0, image.cols);
        int y1 = std::clamp(cvRound(p1.y), 0, image.rows);

        if (x1 <= x0 || y1 <= y0) {
            throw std::runtime_error("Invalid palm crop");
        }

        cv::Mat crop = image(cv::Rect(x0, y0, x1-x0, y1-y0)).clone();

        // Keep coordinates consistent with the clipped crop.
        p0 = cv::Point2f(static_cast<float>(x0), static_cast<float>(y0));
        p1 = cv::Point2f(static_cast<float>(x1), static_cast<float>(y1));

        int side = forRotation
            ? static_cast<int>(std::hypot(crop.rows, crop.cols))
            : std::max(crop.rows, crop.cols);

        int padW = side - crop.cols;
        int padH = side - crop.rows;
        int left = padW / 2;
        int top = padH / 2;

        cv::Mat padded;
        cv::copyMakeBorder(crop, padded, top, padH-top, left, padW-left, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

        bias = cv::Point2i(x0-left, y0-top);
        return padded;
    }

    static cv::Mat preprocess(
        const cv::Mat& frame,
        const std::array<float,18>& palm,
        cv::Point2i& padBias,
        cv::Rect2f& rotatedPalmBox,
        cv::Matx23f& rotation) {

        cv::Point2f p0(palm[0], palm[1]);
        cv::Point2f p1(palm[2], palm[3]);
        cv::Point2i firstBias;

        cv::Mat crop = cropAndPad(frame, p0, p1, true, firstBias);

        cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
        padBias = firstBias;

        cv::Point2f palmPoints[7];
        for (int i = 0; i < 7; ++i) {
            palmPoints[i] = cv::Point2f(palm[4 + i*2] - padBias.x, palm[5 + i*2] - padBias.y);
        }

        cv::Point2f pBase = palmPoints[0];
        cv::Point2f middleBase = palmPoints[2];

        float radians = static_cast<float>(CV_PI / 2 - std::atan2(-(middleBase.y-pBase.y), middleBase.x-pBase.x));

        radians -= 2.0f * static_cast<float>(CV_PI) * std::floor((radians + static_cast<float>(CV_PI)) / (2.0f * static_cast<float>(CV_PI)));

        float angle = radians * 180.0f / static_cast<float>(CV_PI);

        // Palm box coordinates in the padded crop.
        p0 -= cv::Point2f(static_cast<float>(padBias.x), static_cast<float>(padBias.y));
        p1 -= cv::Point2f(static_cast<float>(padBias.x), static_cast<float>(padBias.y));

        // The first crop's bias has already been applied to landmarks.
        // Rotate around the crop center as in the reference implementation.
        cv::Point2f center((p0.x+p1.x)*0.5f, (p0.y+p1.y)*0.5f);

        rotation = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat rotated;
        cv::warpAffine(crop, rotated, rotation, crop.size());

        cv::Point2f rp0(1e9f, 1e9f);
        cv::Point2f rp1(-1e9f, -1e9f);

        for (const auto& point : palmPoints) {
            
            cv::Point2f q(rotation(0,0)*point.x + rotation(0,1)*point.y + rotation(0,2), rotation(1,0)*point.x + rotation(1,1)*point.y + rotation(1,2));

            rp0.x = std::min(rp0.x, q.x);
            rp0.y = std::min(rp0.y, q.y);
            rp1.x = std::max(rp1.x, q.x);
            rp1.y = std::max(rp1.y, q.y);
        }

        cv::Point2i secondBias;
        cv::Mat handCrop = cropAndPad(rotated, rp0, rp1, false, secondBias);

        rotatedPalmBox = cv::Rect2f(rp0.x, rp0.y, rp1.x-rp0.x, rp1.y-rp0.y);

        cv::resize(handCrop, handCrop, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_AREA);

        handCrop.convertTo(handCrop, CV_32FC3, 1.0 / 255.0);

        // The official implementation returns a normalized RGB NHWC image.
        // ConvertTo preserves the 3-channel layout.
        return handCrop;
    }

    static HandPose decodeLandmarks(
        const float* screen,
        const float* world,
        float confidence,
        float handedness,
        const cv::Rect2f& rotatedPalmBox,
        const cv::Point2i& padBias,
        const cv::Matx23f& rotation) {

        HandPose hand;
        hand.confidence = confidence;
        hand.handedness = handedness;

        float scale = std::max(rotatedPalmBox.width, rotatedPalmBox.height) / INPUT_SIZE;

        // Invert the rotation transform.
        cv::Matx22f invR(rotation(0,0), rotation(1,0), rotation(0,1), rotation(1,1));

        cv::Point2f translation(rotation(0,2), rotation(1,2));
        cv::Point2f invT(-(invR(0,0)*translation.x + invR(0,1)*translation.y), -(invR(1,0)*translation.x + invR(1,1)*translation.y));

        cv::Point2f center(rotatedPalmBox.x + rotatedPalmBox.width * 0.5f, rotatedPalmBox.y + rotatedPalmBox.height * 0.5f);

        cv::Point2f originalCenter(invR(0,0)*center.x + invR(0,1)*center.y + invT.x, invR(1,0)*center.x + invR(1,1)*center.y + invT.y);

        float minX = 1e9f, minY = 1e9f;
        float maxX = -1e9f, maxY = -1e9f;

        for (int i = 0; i < 21; ++i) {
            float x = (screen[i*3] - INPUT_SIZE*0.5f) * scale;
            float y = (screen[i*3+1] - INPUT_SIZE*0.5f) * scale;
            float z = screen[i*3+2] * scale;

            cv::Point2f p(rotation(0,0)*x + rotation(1,0)*y, rotation(0,1)*x + rotation(1,1)*y);

            hand.landmarks[i] = cv::Point3f(p.x + originalCenter.x + padBias.x, p.y + originalCenter.y + padBias.y, z);

            hand.worldLandmarks[i] = cv::Point3f(world[i*3], world[i*3+1], world[i*3+2]);

            minX = std::min(minX, hand.landmarks[i].x);
            minY = std::min(minY, hand.landmarks[i].y);
            maxX = std::max(maxX, hand.landmarks[i].x);
            maxY = std::max(maxY, hand.landmarks[i].y);
        }

        // Approximate hand box from landmarks.
        hand.box = cv::Rect2f(minX, minY, maxX-minX, maxY-minY);
        return hand;
    }
};
