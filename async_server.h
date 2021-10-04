#pragma once
#include <memory>
#include <iostream>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include "ocellus_ml_service.grpc.pb.h"
#include "inference.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using ocellus::InferRequest;
using ocellus::InferResponse;
using ocellus::OcellusMLService;

namespace ocellus {
    class ServerImpl final {
    public:
        ~ServerImpl() {
            server_->Shutdown();
            // Always shutdown the completion queue after the server.
            cq_->Shutdown();
        }

        // There is no shutdown handling in this code.
        void Run(std::string port) {
            std::string server_address("0.0.0.0:" + port);

            ServerBuilder builder;
            // Listen on the given address without any authentication mechanism.
            builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
            // Register "service_" as the instance through which we'll communicate with
            // clients. In this case it corresponds to an *asynchronous* service.
            builder.RegisterService(&service_);
            // Get hold of the completion queue used for the asynchronous communication
            // with the gRPC runtime.
            cq_ = builder.AddCompletionQueue();
            // Finally assemble the server.
            server_ = builder.BuildAndStart();
            std::cout << "Async-Server listening on " << server_address << std::endl;

            // Proceed to the server's main loop.
            HandleRpcs();
        }

    private:
        class CallData {
        public:
            virtual void Proceed() = 0;
        };

        // Class encompasing the state and logic needed to serve a request.
        class InferCallData final : public CallData {
        public:
            // Take in the "service" instance (in this case representing an asynchronous
            // server) and the completion queue "cq" used for asynchronous communication
            // with the gRPC runtime.
            explicit InferCallData(OcellusMLService::AsyncService* service, ServerCompletionQueue* cq)
                : responder_(&ctx_), service_(service), cq_(cq), status_(CREATE) {
                // Invoke the serving logic right away.
                Proceed();
            }

            void Proceed() {
                if (status_ == CREATE) {
                    // Make this instance progress to the PROCESS state.
                    status_ = PROCESS;

                    // As part of the initial CREATE state, we *request* that the system
                    // start processing Init requests. In this request, "this" acts are
                    // the tag uniquely identifying the request (so that different CallData
                    // instances can serve different requests concurrently), in this case
                    // the memory address of this CallData instance.
                    service_->RequestInfer(&ctx_, &request_, &responder_, cq_, cq_,
                        this);
                }
                else if (status_ == PROCESS) {
                    // Spawn a new CallData instance to serve new clients while we process
                    // the one for this CallData. The instance will deallocate itself as
                    // part of its FINISH state.
                    new InferCallData(service_, cq_);

                    // The actual processing.
                    std::string serialized_buf = request_.imagedata();
                    std::string err = inference::Infer(
                        request_.modelfile(),
                        serialized_buf.c_str(),
                        request_.width(),
                        request_.height(),
                        request_.minscore(),
                        request_.enabledlabels(),
                        &reply_
                    );

                    if (err.empty() == false)
                    {
                        std::cerr << err << std::endl;
                        reply_.add_errors(err);
                    }

                    // And we are done! Let the gRPC runtime know we've finished, using the
                    // memory address of this instance as the uniquely identifying tag for
                    // the event.
                    status_ = FINISH;
                    responder_.Finish(reply_, Status::OK, this);
                }
                else {
                    GPR_ASSERT(status_ == FINISH);
                    // Once in the FINISH state, deallocate ourselves (CallData).
                    delete this;
                }
            }

        private:
            // The means to get back to the client.
            ServerAsyncResponseWriter<InferResponse> responder_;
            // What we get from the client.
            InferRequest request_;
            // What we send back to the client.
            InferResponse reply_; // The means of communication with the gRPC runtime for an asynchronous
            // server.
            OcellusMLService::AsyncService* service_;
            // The producer-consumer queue where for asynchronous server notifications.
            ServerCompletionQueue* cq_;
            // Context for the rpc, allowing to tweak aspects of it such as the use
            // of compression, authentication, as well as to send metadata back to the
            // client.
            ServerContext ctx_;

            // Let's implement a tiny state machine with the following states.
            enum CallStatus { CREATE, PROCESS, FINISH };
            CallStatus status_;  // The current serving state.
        };

        // This can be run in multiple threads if needed.
        void HandleRpcs() {
            // Spawn a new CallData instance to serve new clients.
            new InferCallData(&service_, cq_.get());
            void* tag;  // uniquely identifies a request.
            bool ok;
            while (true) {
                // Block waiting to read the next event from the completion queue. The
                // event is uniquely identified by its tag, which in this case is the
                // memory address of a CallData instance.
                // The return value of Next should always be checked. This return value
                // tells us whether there is any kind of event or cq_ is shutting down.
                GPR_ASSERT(cq_->Next(&tag, &ok));
                //GPR_ASSERT(ok);
                if (!ok) {
                    std::cerr << "Error processing request" << std::endl;
                }
                static_cast<CallData*>(tag)->Proceed();
            }
        }

        std::unique_ptr<ServerCompletionQueue> cq_;
        OcellusMLService::AsyncService service_;
        std::unique_ptr<Server> server_;
    };
}
