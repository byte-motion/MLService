#include "async_server.h"
#include "sync_server.h"


int main(int argc, char** argv) {
    std::string port = argc > 1 ? std::string(argv[1]) : "50055";

    if (argc > 2 && argv[2] == "async") {
        ocellus::ServerImpl server;
        server.Run(port);
    }
    else {
        ocellus::OcellusMLServiceImpl service;
        service.Run(port);
    }

    return 0;
};
