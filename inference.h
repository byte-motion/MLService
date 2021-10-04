
#ifndef INFERENCE_H
#define INFERENCE_H

#include "ocellus_ml_service.grpc.pb.h"
#include <google/protobuf/repeated_field.h>
#include <string>

namespace ocellus
{
namespace inference
{
const std::string Infer(const std::string &, const char *, const int, const int, const float,
                        const google::protobuf::RepeatedField<google::protobuf::int32> &, ocellus::InferResponse *);

const std::string Unload(const std::string &);
} // namespace inference
} // namespace ocellus

#endif // INFERENCE_H
