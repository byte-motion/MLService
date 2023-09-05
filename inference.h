
#ifndef INFERENCE_H
#define INFERENCE_H

#include "ml_service.grpc.pb.h"
#include <google/protobuf/repeated_field.h>
#include <string>

namespace byte_motion
{
    namespace inference
    {
        const std::string Infer(const std::string &, const char *, const int, const int, const float,
                                const google::protobuf::RepeatedField<google::protobuf::int32> &, byte_motion::InferResponse *);

        const std::string Unload(const std::string &);
    } // namespace inference
} // namespace byte_motion

#endif // INFERENCE_H
