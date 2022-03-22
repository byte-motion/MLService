#include "inference.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/script.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdlib.h>
#include <string>

// #define DEBUG

using namespace std;
using namespace caffe2;

namespace ocellus
{
    namespace inference
    {
        std::mutex mtx;
        std::map<string, torch::jit::Module> modelMap;

        const void terminate(std::string msg)
        {
            std::ofstream logfile;
            const auto now = std::chrono::system_clock::now().time_since_epoch();
            const auto ts = std::chrono::duration_cast<std::chrono::seconds>(now).count();
            logfile.open("/mnt/ocellus/ocellus_ml_crash" + std::to_string(ts) + ".log");
            logfile << msg;
            logfile.close();
            cerr << msg << endl;
            exit(EXIT_FAILURE);
        }

        const string getRelPath(const string &absPath)
        {
            const size_t lastSlash = absPath.find_last_of("/\\");
            const string subStr = absPath.substr(0, lastSlash);
            const size_t secondLastSlash = subStr.find_last_of("/\\");

            return absPath.substr(secondLastSlash + 1);
        }

        const string Infer(const string &modelFileAbsPath, const char *matData, const int width, const int height,
                           const float minScore, const google::protobuf::RepeatedField<google::protobuf::int32> &enabledLabels,
                           ocellus::InferResponse *reply)
        {
            try
            {
                torch::jit::Module module;
                // Compare relative path from extracted tarball location
                auto modelFile = getRelPath(modelFileAbsPath);
                {
                    std::lock_guard<std::mutex> lck(mtx);
                    if (modelMap.count(modelFile) == 0)
                    {
                        c10::cuda::CUDACachingAllocator::emptyCache();
                        cout << "Pytorch File: " << modelFile << endl;
                        torch::autograd::AutoGradMode guard(false);
                        module = torch::jit::load(modelFileAbsPath);
                        modelMap[modelFile] = module;

                        assert(module.buffers().size() > 0);
                    }
                    else
                    {
                        module = modelMap[modelFile];
                    }
                }

                auto start_time = chrono::high_resolution_clock::now();
#ifdef DEBUG
                cout << "Inference requested: " << width << "x" << height << " @" << minScore << endl;
#endif

                // Assume that the entire model is on the same device.
                // We just put input to this device.
                c10::Device device = (*begin(module.buffers())).device();

                // FPN models require divisibility of 32
                assert(height % 32 == 0 && width % 32 == 0);
                const int batch = 1;
                const int channels = 3;
                auto input = torch::from_blob((void *)matData, {1, height, width, channels}, torch::kUInt8);

                // NHWC to NCHW
                input = input.to(device, torch::kFloat).permute({0, 3, 1, 2}).contiguous();

                array<float, 3> im_info_data{height * 1.0f, width * 1.0f, 1.0f};
                auto im_info = torch::from_blob(im_info_data.data(), {1, 3}).to(device);

                // run the network
                auto output = module.forward({make_tuple(input, im_info)});
                if (device.is_cuda())
                {
                    c10::cuda::getCurrentCUDAStream().synchronize();
                }

                auto outputs = output.toTuple()->elements();

                // parse Mask R-CNN outputs, convert to CPU types from CUDA types
                auto bbox = outputs[0].toTensor().cpu();   // .flatten(0).contiguous().cpu();
                auto scores = outputs[1].toTensor().cpu(); // .flatten(0).contiguous().cpu();
                auto labels = outputs[2].toTensor().cpu(); // .flatten(0).contiguous().cpu();
                auto mask_probs = outputs[3].toTensor().cpu();

                int num_instances = bbox.sizes()[0];

#ifdef DEBUG
                cout << "bbox: " << bbox.toString() << " " << bbox.sizes() << endl;
                cout << "scores: " << scores.toString() << " " << scores.sizes() << endl;
                cout << "labels: " << labels.toString() << " " << labels.sizes() << endl;
                cout << "mask_probs: " << mask_probs << " " << mask_probs.sizes() << endl;
                cout << "num_instances: " << num_instances << endl;
#endif

                // Get data pointers for iteration
                float *bbox_ptr = bbox.data_ptr<float>();
                float *scores_ptr = scores.data_ptr<float>();
                float *labels_ptr = labels.data_ptr<float>();

                for (int i = 0; i < num_instances; ++i)
                {
                    int label = (int)labels_ptr[i];
                    float score = scores_ptr[i];
                    const int *labelPos = find(begin(enabledLabels), end(enabledLabels), label);
                    if (labelPos == end(enabledLabels) || score < minScore)
                    {
                        continue;
                    }
                    float *box = bbox_ptr + i * 4;

#ifdef DEBUG
                    // xy1, xy2, width, height
                    cout << "Prediction " << i << "/" << num_instances << ", xyxy=(";
                    cout << box[0] << ", " << box[1] << ", " << box[2] << ", " << box[3] << "); score=" << score
                         << "; label=" << label << endl;
#endif

                    // contour (square), resize to width and height of bounding box
                    // Position is xy1
                    auto maskSlice = mask_probs[i][label];
                    const auto maskSize = maskSlice.size(0) * maskSlice.size(1);
                    const float *mask = maskSlice.data_ptr<float>();

                    auto *result = reply->add_result();

                    result->add_bbox(box[0]);
                    result->add_bbox(box[1]);
                    result->add_bbox(box[2]);
                    result->add_bbox(box[3]);
                    result->set_score(score);
                    result->set_label(label);

                    for (int j = 0; j < maskSize; ++j)
                    {
                        result->add_mask(mask[j]);
                    }

#ifdef DEBUG
                    // xy1, xy2, width, height
                    cout << "Mask size " << result->mask_size() << endl;
                    // // save the 28x28 mask
                    // cv::Mat cv_mask(28, 28, CV_32FC1);
                    // memcpy(cv_mask.data, mask, 28 * 28 * sizeof(float));
                    // cv::imwrite("mask" + to_string(i) + ".png", cv_mask * 255.);
#endif
                }

                auto end_time = chrono::high_resolution_clock::now();
                auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000;
                cout << "Prediction Result Count: " << reply->result_size()
                     << ". Latency (should vary with different inputs): " << ms << "ms (" << modelFile << ")" << endl;
            }
            catch (const exception &ex)
            {
                terminate(ex.what());
            }
            catch (...)
            {
                terminate("Abnormal termination");
            }

            return "";
        }
        const string Unload(const string &modelFileAbsPath)
        {
            try
            {
                auto modelFile = getRelPath(modelFileAbsPath);
                {
                    std::lock_guard<std::mutex> lck(mtx);
                    if (modelMap.count(modelFile) > 0)
                    {
                        // Deconstructor called after erased from map
                        auto erased = modelMap.erase(modelFile);
                        cout << "Unloaded (" << erased << "): " << modelFile << endl;
                    }
                    else
                    {
                        string err = "Could not unload  " + modelFile + ". Model not loaded";
                        cerr << err << endl;
                        return err;
                    }
                }
            }
            catch (const exception &ex)
            {
                terminate(ex.what());
            }
            catch (...)
            {
                terminate("Abnormal termination");
            }

            return "";
        }
    } // namespace inference
} // namespace ocellus
