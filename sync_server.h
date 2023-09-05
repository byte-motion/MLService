#pragma once
#include <iostream>
#include <memory>
#include <string>

#include "inference.h"
#include "ml_service.grpc.pb.h"
#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

using byte_motion::InferRequest;
using byte_motion::InferResponse;
using byte_motion::MLService;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace byte_motion
{
    // Logic and data behind the server's behavior.
    class SyncServiceImpl final : public MLService::Service
    {
    public:
        void Run(std::string port)
        {
            std::string server_address("0.0.0.0:" + port);

            ServerBuilder builder;
            // Listen on the given address without any authentication mechanism.
            builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
            // Register "service" as the instance through which we'll communicate with
            // clients. In this case it corresponds to an *synchronous* service.
            builder.RegisterService(this);
            // Finally assemble the server.
            std::unique_ptr<Server> server(builder.BuildAndStart());
            std::cout << "Sync-Server listening on " << server_address << std::endl;

            // Wait for the server to shutdown. Note that some other thread must be
            // responsible for shutting down the server for this call to ever return.
            server->Wait();
        }

        Status Infer(ServerContext *context, const InferRequest *request, InferResponse *response) override
        {
            // The actual processing.
            std::string serialized_buf = request->imagedata();
            std::string err = inference::Infer(request->modelfile(), serialized_buf.c_str(), request->width(),
                                               request->height(), request->minscore(), request->enabledlabels(), response);

            if (err.empty() == false)
            {
                std::cerr << err << std::endl;
                response->add_errors(err);
            }

            return Status::OK;
        }

        Status Unload(ServerContext *context, const UnloadRequest *request, UnloadResponse *response) override
        {
            std::string err = inference::Unload(request->modelfile());

            if (err.empty() == false)
            {
                std::cerr << err << std::endl;
                response->add_errors(err);
            }

            return Status::OK;
        }
    };
} // namespace byte_motion
